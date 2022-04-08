#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2022/1/19 11:52 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 11:52   wangfc      1.0         None
"""



DEFAULT_LOG_LEVEL = "INFO"
ENV_LOG_LEVEL = "LOG_LEVEL"
LOG_FORMAT = '%(asctime)s %(filename)s %(module)s %(funcName)s [line:%(lineno)d] [%(levelname)s] %(message)s'



STANDARD_QUESTION = 'standard_question'
EXTEND_QUESTION = 'extend_question'
SCENARIO = 'scenario'
INTENT_CATEGORY = 'intent_category'
FAQ = 'faq'
CHAT= 'chat'


ID = 'id'
QUESTION = 'question'
QUESTION_LENGTH = 'question_length'
INTENT = 'intent'
# 新增 SUB_INTENT 和 SUB_INTENT_RANKING_KEY
SUB_INTENT = 'sub_intent'
SUB_INTENT_RANKING_KEY = 'sub_intent_ranking'
RESPONSE = 'response'
SOURCE = 'source'

# 针对 ivr 数据增加一列： 每个标准问  function_name
FUNCTION_NAME = "function_name"
# intent + entity 对应一个 response_intent
RESPONSE_INTENT = 'response_intent'

# @unique
# class DATATYPE(Enum):
RAW_DATATYPE = 'raw_data' # 原始数据
ALL_DATATYPE = 'all_data' # 训练模型的全部数据
TRAIN_DATATYPE = 'train'
EVAL_DATATYPE = 'dev'
TEST_DATATYPE = 'test'

DEFAULT_TRAINING_DATA_OUTPUT_PATH = "training_data.yml"

TF_INPUT_EXAMPLE_LABEL_COLUMN = "label_id"

RANDOM_BATCH_STRATEGY = 'random_batch_strategy'
BALANCED_BATCH_STRATEGY = 'balanced_batch_strategy'
SEQUENCE_BATCH_STRATEGY = 'sequence_batch_strategy'




TEXT = 'text'
RAW_TEXT = 'raw_text'
SUB_LABEL_TEXT = 'sub_label_text'
INTENT_AND_ATTRIBUTE = "intent_and_attribute"
INTENT_NAME_KEY = "name"
# retrieval intent with the response key suffix attached : 如 'faq/如何查询自助开户进度'
INTENT_RESPONSE_KEY = "intent_response_key"
RESPONSE_IDENTIFIER_DELIMITER = "/"

ACTION_TEXT = "action_text"
ACTION_NAME = "action_name"

METADATA = "metadata"
METADATA_INTENT = "intent"
METADATA_EXAMPLE = "example"

INTENT_RANKING_KEY = "intent_ranking"

PREDICTED_CONFIDENCE_KEY = "confidence"


BILOU_ENTITIES = "bilou_entities"
BILOU_ENTITIES_ROLE = "bilou_entities_role"
BILOU_ENTITIES_GROUP = "bilou_entities_group"


CLASSIFIER = 'classifier'
EXTRACTOR = "extractor"

TRAINABLE_EXTRACTORS = {"MitieEntityExtractor", "CRFEntityExtractor", "DIETClassifier"}


ENTITIES = "entities"
ENTITY_TAGS = "entity_tags"
ENTITY_ATTRIBUTE_TYPE = "entity"
ENTITY_ATTRIBUTE_GROUP = "group"
ENTITY_ATTRIBUTE_ROLE = "role"
ENTITY_ATTRIBUTE_VALUE = "value"
ENTITY_ATTRIBUTE_START = "start"
ENTITY_ATTRIBUTE_END = "end"
NO_ENTITY_TAG = "O"



ENTITY_ATTRIBUTE_TEXT = "text"
ENTITY_ATTRIBUTE_CONFIDENCE = "confidence"

ENTITY_ATTRIBUTE_CONFIDENCE_TYPE = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{ENTITY_ATTRIBUTE_TYPE}"
)
ENTITY_ATTRIBUTE_CONFIDENCE_GROUP = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{ENTITY_ATTRIBUTE_GROUP}"
)
ENTITY_ATTRIBUTE_CONFIDENCE_ROLE = (
    f"{ENTITY_ATTRIBUTE_CONFIDENCE}_{ENTITY_ATTRIBUTE_ROLE}"
)


SPLIT_ENTITIES_BY_COMMA = "split_entities_by_comma"
SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE = True
SINGLE_ENTITY_ALLOWED_INTERLEAVING_CHARSET = {".", ",", " ", ";"}






COMPONENT_INDEX = "index"

DIAGNOSTIC_DATA = "diagnostic_data"


MESSAGE_ATTRIBUTES = [
    TEXT,
    INTENT,
    RESPONSE,
    ACTION_NAME,
    ACTION_TEXT,
    INTENT_RESPONSE_KEY,
]

TOKENS_NAMES = {
    TEXT: "text_tokens",
    INTENT: "intent_tokens",
    RESPONSE: "response_tokens",
    ACTION_NAME: "action_name_tokens",
    ACTION_TEXT: "action_text_tokens",
    INTENT_RESPONSE_KEY: "intent_response_key_tokens",
}


FEATURE_TYPE_SENTENCE = "sentence"
FEATURE_TYPE_SEQUENCE = "sequence"

# the dense featurizable attributes are essentially text attributes
DENSE_FEATURIZABLE_ATTRIBUTES = [
    TEXT,
    RESPONSE,
    ACTION_TEXT,
]

FEATURIZER_CLASS_ALIAS = "alias"

MIN_ADDITIONAL_REGEX_PATTERNS = 10
MIN_ADDITIONAL_CVF_VOCABULARY = 1000


# attribute 对应的 language_model_doc 名称，其保存在 message 的属性中
LANGUAGE_MODEL_DOCS = {
    TEXT: "text_language_model_doc",
    RESPONSE: "response_language_model_doc",
    ACTION_TEXT: "action_text_model_doc",
}

NUMBER_OF_SUB_TOKENS = "number_of_sub_tokens"
NO_LENGTH_RESTRICTION = -1


# How many labels are at max put into the output
# ranking, everything else will be cut off
LABEL_RANKING_LENGTH = 10



DEFAULT_CORE_SUBDIRECTORY_NAME = "core"
DEFAULT_NLU_SUBDIRECTORY_NAME = "nlu"

DEFAULT_MODELS_PATH ='models'
NLU_MODEL_NAME_PREFIX = "nlu_"



INTENT_TO_OUTPUT_MAPPING_FILENAME = "intent_to_output_mapping"
INTENT_ATTRIBUTE_MAPPING_FILENAME = "intent_attribute_mapping.json"
INTENT_MAPPING_KEY = "intent"
ATTRIBUTE_MAPPING_KEY = "attribute"
INTENT_TO_ATTRIBUTE_MAPPING = "intent_to_attribute_mapping"



MODEL_CONFIG_SCHEMA_FILE = "shared/utils/schemas/model_config.yml"
CONFIG_SCHEMA_FILE = "shared/utils/schemas/config.yml"
RESPONSES_SCHEMA_FILE = "shared/nlu/training_data/schemas/responses.yml"
SCHEMA_EXTENSIONS_FILE = "shared/utils/pykwalify_extensions.py"


DEFAULT_REQUEST_TIMEOUT = 60 * 5  # 5 minutes
DEFAULT_RESPONSE_TIMEOUT = 60 * 60  # 1 hour

