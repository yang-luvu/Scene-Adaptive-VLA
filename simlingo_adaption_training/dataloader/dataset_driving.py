"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""

import os
import ujson
import json
import numpy as np
import random
import cv2
import re
import gzip

import torch
from simlingo_adaption_training.utils.custom_types import DatasetOutput
from simlingo_adaption_training.dataloader.dataset_base import BaseDataset


VIZ_DATA = False

class Data_Driving(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(self,
            **cfg,
        ):
        super().__init__(dreamer=False, **cfg)
        self._warned_commentary_templates = set()
        self._warned_qa_questions = set()
        self._warned_qa_answers = set()
        self._warned_qa_answer_placeholders = set()
        self._qa_answer_aug_lookup = 0
        self._qa_answer_aug_hit = 0
        self._qa_answer_aug_inline_fallback = 0
        self._qa_answer_aug_default_fallback = 0
        self._qa_locations = [
            'nearby to the front of the ego vehicle',
            'nearby to the front right of the ego vehicle',
            'nearby to the front left of the ego vehicle',
            'nearby on the left side of the ego vehicle',
            'nearby to the front of it',
            'nearby to the front right of it',
            'nearby to the front left of it',
            'nearby on its left side',
            'far to the front left of the ego vehicle',
            'far to the front right of the ego vehicle',
            'far to the front of the ego vehicle',
            'far to the left side of the ego vehicle',
            'far to the right side of the ego vehicle',
            'far to the front left of it',
            'far to the front right of it',
            'far to the front of it',
            'far to its left side',
            'far to its right side',
            'to the front of the ego vehicle',
            'to the front right of the ego vehicle',
            'to the front left of the ego vehicle',
            'to the front of it',
            'to the front right of it',
            'to the front left of it',
            'on the left side of the ego vehicle',
            'on the right side of the ego vehicle',
            'on its left side',
            'on its right side',
            # Keep these at the end so longer phrases are replaced first.
            'nearby',
            'far',
        ]

    @staticmethod
    def _parse_frame_id(measurement_path: str):
        file_name = os.path.basename(measurement_path)
        if file_name.endswith(".json.gz"):
            file_name = file_name[:-8]
        else:
            file_name = os.path.splitext(file_name)[0]
        try:
            return int(file_name)
        except ValueError:
            return None

    @staticmethod
    def _normalize_template_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.replace(' ,', ',').replace(' .', '.')
        return text.strip()

    @staticmethod
    def _build_object_aliases(object_type: str):
        aliases = [object_type]
        alias_rules = [
            (" SUV", " car"),
            (" SUVs", " cars"),
            (" suv", " car"),
            (" suvs", " cars"),
            (" car", " SUV"),
            (" cars", " SUVs"),
        ]
        for source, target in alias_rules:
            if source in object_type:
                aliases.append(object_type.replace(source, target))
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(aliases))

    def _replace_objects_with_placeholder(self, text: str, objects):
        replaced_objects = []
        for object_type in objects:
            for alias in self._build_object_aliases(object_type):
                if alias and alias in text:
                    text = text.replace(alias, '<OBJECT>')
                    replaced_objects.append(object_type)
                    break
        return text, replaced_objects

    @staticmethod
    def _build_inline_answer_augmentations(answer_template: str):
        plural_prefix = "The important objects in the scene are "
        singular_prefix = "The important object in the scene is "
        body = None
        templates = []

        if answer_template.startswith(plural_prefix):
            body = answer_template[len(plural_prefix):].rstrip('.')
            templates = [
                f"The important objects in the scene are {body}.",
                f"In the scene, the important objects are {body}.",
                f"The key objects in the scene are {body}.",
            ]
        elif answer_template.startswith(singular_prefix):
            body = answer_template[len(singular_prefix):].rstrip('.')
            templates = [
                f"The important object in the scene is {body}.",
                f"In the scene, the important object is {body}.",
                f"The key object in the scene is {body}.",
            ]

        if not templates:
            return []

        return [Data_Driving._normalize_template_text(candidate) for candidate in templates]

    def __getitem__(self, index):
        """Returns the item at index idx. """
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)

        data = {}
        images = self.images[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        augment_exists = self.augment_exists[index]

        ######################################################
        ######## load current and future measurements ########
        ######################################################
        loaded_measurements, current_measurement, measurement_file_current = self.load_current_and_future_measurements(
            measurements,
            sample_start
            )
        
        data['measurement_path'] = measurement_file_current
        frame_id = self._parse_frame_id(measurement_file_current)

        ego_xy = None
        ego_yaw = None
        ego_matrix = current_measurement.get("ego_matrix")
        if ego_matrix is not None:
            ego_matrix_np = np.asarray(ego_matrix, dtype=np.float32)
            if ego_matrix_np.ndim == 2 and ego_matrix_np.shape[0] >= 3 and ego_matrix_np.shape[1] >= 4:
                ego_xy = (float(ego_matrix_np[0, 3]), float(ego_matrix_np[1, 3]))
                ego_yaw = float(np.arctan2(ego_matrix_np[1, 0], ego_matrix_np[0, 0]))

        timestamp = None
        for key in ("timestamp", "time", "sim_time", "frame_time"):
            if key in current_measurement and current_measurement[key] is not None:
                try:
                    timestamp = float(current_measurement[key])
                    break
                except (TypeError, ValueError):
                    continue
        if timestamp is None and frame_id is not None:
            timestamp = float(frame_id) * 0.05

        # Determine whether the augmented camera or the normal camera is used.
        if augment_exists and random.random() <= self.img_shift_augmentation_prob and self.img_shift_augmentation:
            augment_sample = True
            aug_rotation = current_measurement['augmentation_rotation']
            aug_translation = current_measurement['augmentation_translation']
        else:
            augment_sample = False
            aug_rotation = 0.0
            aug_translation = 0.0


        ######################################################
        ################## load waypoints ####################
        ######################################################
        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation)
       
        speed_rounded = round(current_measurement['speed'], 1)
        data['speed'] = current_measurement['speed']

        data = self.load_route(data, current_measurement, aug_translation, aug_rotation)

        target_point = np.array(current_measurement['target_point'])
        target_point = self.augment_target_point(target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)
        next_target_point = np.array(current_measurement['target_point_next'])
        next_target_point = self.augment_target_point(next_target_point, y_augmentation=aug_translation, yaw_augmentation=aug_rotation)

        ######################################################
        ################## get commentary & qa ##################
        ######################################################
        commentary_exists = False
        commentary = ''
        if self.use_commentary:
            commentary_file_path = self._resolve_sidecar_path_from_measurement(
                measurement_file_current,
                sidecar_kind="commentary",
            )
            # do not use evaluation routes!!!
            if 'validation_' in commentary_file_path:
                commentary_exists = False
            else:
                try:
                    with gzip.open(commentary_file_path, 'rt') as f:
                        commentary_file = ujson.load(f)
                        commentary_exists = True
                except (FileNotFoundError, ujson.JSONDecodeError):
                    commentary_exists = False
                    commentary_file = None

                if commentary_file is not None:

                    commentary = commentary_file['commentary']
                    # we only augment in 60% of the cases and use the default commentary in 40% of the cases
                    # augmentation is used to increase generalization to a broader set of sentences
                    # but we do not want to overfit to the augmented sentences
                    if self.commentary_augmentation and random.random() < 0.6:
                        if commentary_file['commentary_template'] in self.templates_commentary:
                            commentary = random.choice(self.templates_commentary[commentary_file['commentary_template']])
                            for key, value in commentary_file['placeholder'].items():
                                if key in commentary:
                                    commentary = commentary.replace(key, value)
                            # regex check if <OBJECT> or <LOCATION> or any other <> is still in commentary, if so use default commentary_file['commentary']
                            if re.search(r'<.*?>', commentary):
                                warn_key = commentary_file.get("commentary_template", commentary)
                                if warn_key not in self._warned_commentary_templates:
                                    self._warned_commentary_templates.add(warn_key)
                                    print(
                                        f"WARNING: {commentary} contains placeholders that are not replaced. Using default commentary."
                                    )
                                commentary = commentary_file['commentary']

                    commentary = commentary.replace('..', '.')
                    commentary = commentary.replace('in in', 'in')
        
        qa_exists = False
        if self.use_qa:
            qa_path = self._resolve_sidecar_path_from_measurement(
                measurement_file_current,
                sidecar_kind="vqa",
            )
            if 'validation_' in qa_path:
                qa_exists = False
            else:
                try:
                    with gzip.open(qa_path, 'rt') as f:
                        qa = ujson.load(f)
                    qa_exists = True
                except (FileNotFoundError, ujson.JSONDecodeError):
                    qa_exists = False
                    qa = None

            if qa_exists:
                qas = qa['QA']
                qas = [values for values in qas.values()] # list of lists
                qas = [item for sublist in qas for item in sublist] # flatten list
                while True:
                    qa_chosen = random.choice(qas)
                    qa_question = qa_chosen['Q']
                    qa_answer = qa_chosen['A']
                    
                    # TODO: make this nicer!!!
                    if 'There are no pedestrians.' in qa_answer or \
                            'There is no traffic light' in qa_answer or \
                            'There are no pedestrians.' in qa_answer or \
                            'No, the ego vehicle is not affected by a stop sign.' in qa_answer or \
                            'No, the ego vehicle is not affected by a junction.' in qa_answer or \
                            'There is no traffic light affecting the ego vehicle.' in qa_answer or \
                            'There is no stop sign affecting the ego vehicle.' in qa_answer or \
                            'There is no junction affecting the ego vehicle.' in qa_answer or \
                            'It is not possible to tell' in qa_answer or \
                            'There is no reason for the ego vehicle to brake.' in qa_answer:
                        # only keep in 20% of the cases
                        if random.random() < 0.2:
                            break
                    else:
                        break
                
                # we only augment in 60% of the cases and use the default QA in 40% of the cases
                # augmentation is used to increase generalization to a broader set of sentences
                # but we do not want to overfit to the augmented sentences
                if self.qa_augmentation and random.random() < 0.6:
                    qa_question_org = qa_question
                    qa_answer_org = qa_answer
                    objects = [value['Visual_description'] for key, value in qa['key_object_infos'].items()]
                    qa_question, q_objects = self._replace_objects_with_placeholder(qa_question, objects)
                    qa_answer, a_objects = self._replace_objects_with_placeholder(qa_answer, objects)
                    
                    q_location = ''
                    a_location = ''
                    for location in self._qa_locations:
                        if location in qa_question:
                            qa_question = qa_question.replace(location, '<LOCATION>')
                            if q_location == '':
                                q_location = location
                        if location in qa_answer:
                            qa_answer = qa_answer.replace(location, '<LOCATION>')
                            if a_location == '':
                                a_location = location
                        
                    q_distance = re.search(r'in (\d+) m', qa_question)
                    qa_question = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_question)
                    a_distance = re.search(r'in (\d+) m', qa_answer)
                    qa_answer = re.sub(r'in \d+ m', 'in <DISTANCE>', qa_answer)
                    qa_question = self._normalize_template_text(qa_question)
                    qa_answer = self._normalize_template_text(qa_answer)
                    if len(q_objects)==0:
                        q_objects = ['']
                    if len(a_objects)==0:
                        a_objects = ['']
                    
                    # in 40% of the cases we do not augment the question
                    if len(q_objects) > 1 or len(a_objects) > 1 or random.random() < 0.4: 
                        qa_question = qa_question_org
                        qa_answer = qa_answer_org
                    else:
                        if qa_question in self.q_augment:
                            qa_question = random.choice(self.q_augment[qa_question]).replace('<OBJECT>', q_objects[0]).replace('<LOCATION>', q_location)
                            if q_distance:
                                qa_question = qa_question.replace('<DISTANCE>', q_distance.group(1))
                        else:
                            if qa_question_org not in self._warned_qa_questions:
                                self._warned_qa_questions.add(qa_question_org)
                                print(f"WARNING: {qa_question} not in q_augment. Using default question.")
                            qa_question = qa_question_org
                        self._qa_answer_aug_lookup += 1
                        if qa_answer in self.a_augment:
                            self._qa_answer_aug_hit += 1
                            qa_answer = random.choice(self.a_augment[qa_answer]).replace('<OBJECT>', a_objects[0]).replace('<LOCATION>', a_location)
                            if a_distance:
                                qa_answer = qa_answer.replace('<DISTANCE>', a_distance.group(1))
                        else:
                            inline_augmented_answers = self._build_inline_answer_augmentations(qa_answer)
                            if inline_augmented_answers:
                                self._qa_answer_aug_inline_fallback += 1
                                qa_answer = random.choice(inline_augmented_answers).replace('<OBJECT>', a_objects[0]).replace('<LOCATION>', a_location)
                                if a_distance:
                                    qa_answer = qa_answer.replace('<DISTANCE>', a_distance.group(1))
                            else:
                                self._qa_answer_aug_default_fallback += 1
                                if qa_answer_org not in self._warned_qa_answers:
                                    self._warned_qa_answers.add(qa_answer_org)
                                    print(f"WARNING: {qa_answer} not in a_augment. Using default answer.")
                                qa_answer = qa_answer_org

                        if re.search(r'<.*?>', qa_answer):
                            if qa_answer_org not in self._warned_qa_answer_placeholders:
                                self._warned_qa_answer_placeholders.add(qa_answer_org)
                                print(
                                    f"WARNING: {qa_answer} contains placeholders that are not replaced. Using default answer."
                                )
                            qa_answer = qa_answer_org

        ######################################################
        ######## load navigational_conditioning ########
        ######################################################
        target_options, placeholder_values = self.get_navigational_conditioning( data, current_measurement, target_point, next_target_point)
            
        answer = ''

        prompt_random = random.random()
        
        if self.use_commentary and commentary_exists and prompt_random < self.prompt_probabilities['commentary']:
            if random.random() < 0.2: # 20% of the time we give commentary as prompt
                if random.random() < 0.5:
                    prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} {commentary} Predict the waypoints."
                else:
                    prompt = f"Current speed: {speed_rounded} m/s. Command: {commentary} Predict the waypoints."
                answer = f"Waypoints:"
            else:
                # 80% of the time we want to predict commentary
                prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} What should the ego do next?"
                answer = f"{commentary} Waypoints:"
            self.num_sampled_per_type['commentary'] += 1
            
        elif self.use_qa and qa_exists and prompt_random < (self.prompt_probabilities['qa'] + self.prompt_probabilities['commentary']):
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} Q: {qa_question}"
            answer = f"A: {qa_answer}"
            self.num_sampled_per_type['qa'] += 1
            
        else:
            prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} Predict the waypoints."
            answer = f"Waypoints:"
            self.num_sampled_per_type['driving'] += 1

        # recalculate the probabilties after warmup (when more than 1000 samples have been sampled)
        # we do this in case we dont have qa or commentary for every sample otherwise it would lead to undersampling one of those
        if sum(self.num_sampled_per_type.values()) > 10000 and sum(self.num_sampled_per_type.values()) % 10000 == 0:
            # avoid division by zero
            self.prompt_probabilities = {key: 1/(value + 1e-6) for key, value in self.num_sampled_per_type.items()}
            self.prompt_probabilities = {key: value/sum(self.prompt_probabilities.values()) for key, value in self.prompt_probabilities.items()}
            print(f"Prompt probabilities: {self.prompt_probabilities}")
            print(f"Number of samples per type: {self.num_sampled_per_type}")
            if self._qa_answer_aug_lookup > 0:
                qa_answer_hit_rate = self._qa_answer_aug_hit / self._qa_answer_aug_lookup
                qa_answer_inline_rate = self._qa_answer_aug_inline_fallback / self._qa_answer_aug_lookup
                qa_answer_default_rate = self._qa_answer_aug_default_fallback / self._qa_answer_aug_lookup
                print(
                    f"QA answer augmentation stats - lookup: {self._qa_answer_aug_lookup}, "
                    f"hit: {self._qa_answer_aug_hit} ({qa_answer_hit_rate:.2%}), "
                    f"inline_fallback: {self._qa_answer_aug_inline_fallback} ({qa_answer_inline_rate:.2%}), "
                    f"default_fallback: {self._qa_answer_aug_default_fallback} ({qa_answer_default_rate:.2%})"
                )
            
        answer = answer.replace('..', '.')
        prompt = prompt.replace('..', '.')

        ######################################################
        ######## load current and past images ########
        ######################################################
        data = self.load_images(data, images, augment_sample=augment_sample)
        

        conversation_answer = [
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            },
        ]
        conversation_all = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image"},
                ],
            },
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            }
        ]
        
        images = [data['rgb']]

        data_new = DatasetOutput(
            conversation = conversation_all,
            answer = conversation_answer,
            image_ff = data['rgb'],
            image_ff_org_size=data['rgb_org_size'],
            waypoints = data["waypoints"],
            waypoints_1d = data["waypoints_1d"],
            path = data['route_adjusted'],
            target_points = data['target_points'],
            speed = data['speed'],
            placeholder_values = placeholder_values,
            measurement_path = data['measurement_path'],
            dataset = 'driving',
            ego_xy=ego_xy,
            ego_yaw=ego_yaw,
            timestamp=timestamp,
            frame_id=frame_id,
        )
        
        if VIZ_DATA:
            # front image with path and waypoints and commentary
            self.visualise_cameras(data_new, commentary, data['route_adjusted'], data['waypoints'], options=None, prompt=prompt, answer=answer, name="img")
        return data_new


if __name__ == "__main__":
    from hydra import compose, initialize
    from simlingo_adaption_training.config import TrainConfig
    
    # seed all
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    

    initialize(config_path="../config")
    cfg = compose(config_name="config")
    
    cfg.data_module.base_dataset.use_commentary = True
    cfg.data_module.base_dataset.use_qa = True
    cfg.data_module.base_dataset.img_shift_augmentation = False

    print('Test Dataset')
    dataset = Data_Driving(                        
                        split="train",
                        bucket_name='all',
                        **cfg.data_module,
                        **cfg.data_module.base_dataset,
    )

    for i in range(len(dataset)):
        # shuffle
        data = dataset[i]
