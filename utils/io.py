#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: 
@version: 
@desc:  
@time: 2021/3/30 9:32 

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/3/30 9:32   wangfc      1.0         None

"""

import os
import sys
import json
import tarfile
import tempfile
import warnings
import errno
import re
from io import StringIO
from pathlib import Path
from typing import Text, Optional, Any, Type, Dict, Union, List
import pandas as pd
# pip install  ruamel.yaml
import pickle

import zipfile
from io import BytesIO as IOReader
from ruamel import yaml
from ruamel.yaml import RoundTripRepresenter, YAMLError
from ruamel.yaml.constructor import DuplicateKeyError, BaseConstructor, ScalarNode

from constants import NEXT_MAJOR_VERSION_FOR_DEPRECATIONS
from utils.exceptions import YamlSyntaxException


if sys.version_info >= (3, 7):
    from typing import OrderedDict
else:
    # 兼容 py3.6ba
    from collections import OrderedDict

import logging

logger = logging.getLogger(__name__)

DEFAULT_ENCODING = "utf-8"
YAML_VERSION = (1, 2)
YAML_LINE_MAX_WIDTH = 4096

YAML_FILE_EXTENSIONS = [".yml", ".yaml"]
JSON_FILE_EXTENSIONS = [".json"]
MARKDOWN_FILE_EXTENSIONS = [".md"]
TRAINING_DATA_EXTENSIONS = set(
    JSON_FILE_EXTENSIONS + MARKDOWN_FILE_EXTENSIONS + YAML_FILE_EXTENSIONS
)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def wrap_with_color(*args: Any, color: Text) -> Text:
    return color + " ".join(str(s) for s in args) + bcolors.ENDC


def raise_deprecation_warning(
        message: Text,
        warn_until_version: Text = NEXT_MAJOR_VERSION_FOR_DEPRECATIONS,
        docs: Optional[Text] = None,
        **kwargs: Any,
) -> None:
    """
    Thin wrapper around `raise_warning()` to raise a deprecation warning. It requires
    a version until which we'll warn, and after which the support for the feature will
    be removed.
    """
    if warn_until_version not in message:
        message = f"{message} (will be removed in {warn_until_version})"

    # need the correct stacklevel now
    kwargs.setdefault("stacklevel", 3)
    # we're raising a `FutureWarning` instead of a `DeprecationWarning` because
    # we want these warnings to be visible in the terminal of our users
    # https://docs.python.org/3/library/warnings.html#warning-categories
    raise_warning(message, FutureWarning, docs, **kwargs)


def raise_warning(
        message: Text,
        category: Optional[Type[Warning]] = None,
        docs: Optional[Text] = None,
        **kwargs: Any,
) -> None:
    """Emit a `warnings.warn` with sensible defaults and a colored warning msg."""

    original_formatter = warnings.formatwarning

    def should_show_source_line() -> bool:
        if "stacklevel" not in kwargs:
            if category == UserWarning or category is None:
                return False
            if category == FutureWarning:
                return False
        return True

    def formatwarning(
            message: Text,
            category: Optional[Type[Warning]],
            filename: Text,
            lineno: Optional[int],
            line: Optional[Text] = None,
    ) -> Text:
        """Function to format a warning the standard way."""

        if not should_show_source_line():
            if docs:
                line = f"More info at {docs}"
            else:
                line = ""

        formatted_message = original_formatter(
            message, category, filename, lineno, line
        )
        return wrap_with_color(formatted_message, color=bcolors.WARNING)

    if "stacklevel" not in kwargs:
        # try to set useful defaults for the most common warning categories
        if category == DeprecationWarning:
            kwargs["stacklevel"] = 3
        elif category in (UserWarning, FutureWarning):
            kwargs["stacklevel"] = 2

    warnings.formatwarning = formatwarning
    warnings.warn(message, category=category, **kwargs)
    warnings.formatwarning = original_formatter


def list_files(path: Text) -> List[Text]:
    """Returns all files excluding hidden files.

    If the path points to a file, returns the file."""

    return [fn for fn in list_directory(path) if os.path.isfile(fn)]


def list_directory(path: Text) -> List[Text]:
    """Returns all files and folders excluding hidden files.

    If the path points to a file, returns the file. This is a recursive
    implementation returning files in any depth of the path."""

    if not isinstance(path, str):
        raise ValueError(
            f"`resource_name` must be a string type. " f"Got `{type(path)}` instead"
        )

    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        results = []
        for base, dirs, files in os.walk(path, followlinks=True):
            # sort files for same order across runs
            files = sorted(files, key=_filename_without_prefix)
            # add not hidden files
            good_files = filter(lambda x: not x.startswith("."), files)
            results.extend(os.path.join(base, f) for f in good_files)
            # add not hidden directories
            good_directories = filter(lambda x: not x.startswith("."), dirs)
            results.extend(os.path.join(base, f) for f in good_directories)
        return results
    else:
        raise ValueError(f"Could not locate the resource '{os.path.abspath(path)}'.")


def list_subdirectory(path: Text) -> List[Text]:
    """Returns all files and folders excluding hidden files.

    If the path points to a file, returns the file. This is a recursive
    implementation returning files in any depth of the path."""

    if not isinstance(path, str):
        raise ValueError(
            f"`resource_name` must be a string type. " f"Got `{type(path)}` instead"
        )

    if os.path.isfile(path):
        return []
    elif os.path.isdir(path):
        results = []
        for f in os.listdir(path):
            dir = os.path.join(path, f)
            if os.path.isdir(dir):
                results.append(dir)
        return results
    else:
        raise ValueError(f"Could not locate the resource '{os.path.abspath(path)}'.")


def _filename_without_prefix(file: Text) -> Text:
    """Splits of a filenames prefix until after the first ``_``."""
    return "_".join(file.split("_")[1:])


def is_likely_yaml_file(file_path: Union[Text, Path]) -> bool:
    """Check if a file likely contains yaml.

    Arguments:
        file_path: path to the file

    Returns:
        `True` if the file likely contains data in yaml format, `False` otherwise.
    """
    return Path(file_path).suffix in set(YAML_FILE_EXTENSIONS)


def is_likely_json_file(file_path: Text) -> bool:
    """Check if a file likely contains json.

    Arguments:
        file_path: path to the file

    Returns:
        `True` if the file likely contains data in json format, `False` otherwise.
    """
    return Path(file_path).suffix in set(JSON_FILE_EXTENSIONS)


def is_likely_markdown_file(file_path: Text) -> bool:
    """Check if a file likely contains markdown.

    Arguments:
        file_path: path to the file

    Returns:
        `True` if the file likely contains data in markdown format,
        `False` otherwise.
    """
    return Path(file_path).suffix in set(MARKDOWN_FILE_EXTENSIONS)


def markdown_file_extension() -> Text:
    """Return Markdown file extension"""
    return MARKDOWN_FILE_EXTENSIONS[0]


def read_file(filename: Union[Text, Path], encoding: Text = DEFAULT_ENCODING) -> Any:
    """Read text from a file."""

    try:
        with open(filename, encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Failed to read file, " f"'{os.path.abspath(filename)}' does not exist."
        )
    except UnicodeDecodeError:
        raise UnicodeDecodeError(
            f"Failed to read file '{os.path.abspath(filename)}', "
            f"could not read the file using {encoding} to decode "
            f"it. Please make sure the file is stored with this "
            f"encoding."
        )


def read_json_file(json_path) -> Union[Dict, List]:
    """
    修改名册 load_json to read_json_file
    json.load() can deserialize a file itself i.e. it accepts a file object
    json.loads() does not take the file path, but the file contents as a string, s stands for string.
    """
    # with open(json_path, mode='r', encoding='utf-8') as f:
    #     json_object = json.load(f)
    content = read_file(filename=json_path)
    try:
        json_object = json.loads(content)
        logger.info(
            "读取json数据 from {}".format(json_path))
        return json_object

    except ValueError as e:
        raise ValueError(
            f"读取json数据失败，Failed to read json from '{os.path.abspath(json_path)}'. Error: {e}"
        )
    return json_object


def write_text_file(
        content: Text,
        file_path: Union[Text, Path],
        encoding: Text = DEFAULT_ENCODING,
        append: bool = False,
) -> None:
    """Writes text to a file.

    Args:
        content: The content to write.
        file_path: The path to which the content should be written.
        encoding: The encoding which should be used.
        append: Whether to append to the file or to truncate the file.

    """
    mode = "a" if append else "w"
    with open(file_path, mode, encoding=encoding) as file:
        file.write(content)


def json_to_string(obj: Any, **kwargs: Any) -> Text:
    """Dumps a JSON-serializable object to string.

    Args:
        obj: JSON-serializable object.
        kwargs: serialization options. Defaults to 2 space indentation
                and disable escaping of non-ASCII characters.

    Returns:
        The objects serialized to JSON, as a string.
    """
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def dump_obj_as_json_to_file(filename: Union[Text, Path], obj: Any, indent=2) -> None:
    """Dump an object as a json string to a file."""
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    # with open(json_path, mode='w', encoding='utf-8') as f:
    #     json.dump(json_object, f, ensure_ascii=False, indent=4)
    json_string = json_to_string(obj, ensure_ascii=False, indent=indent)
    write_text_file(json_string, filename)
    logger.info("写入Write json_object into {}".format(filename))


def json_unpickle(file_name: Union[Text, Path]) -> Any:
    """Unpickle an object from file using json.

    Args:
        file_name: the file to load the object from

    Returns: the object
    """
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle

    jsonpickle_numpy.register_handlers()

    file_content = read_file(file_name)
    return jsonpickle.loads(file_content)


def json_pickle(file_name: Union[Text, Path], obj: Any) -> None:
    """Pickle an object to a file using json.

    Args:
        file_name: the file to store the object to
        obj: the object to store
    """
    import jsonpickle.ext.numpy as jsonpickle_numpy
    import jsonpickle

    jsonpickle_numpy.register_handlers()

    write_text_file(jsonpickle.dumps(obj), file_name)


def load_stopwords(path) -> List:
    if path and os.path.exists(path):
        stopwords_set = set()
        with open(path, mode='r', encoding='utf-8') as f:
            for word in f.readlines():
                word = word.strip().lower()
                stopwords_set.add(word)
        stopwords_ls = sorted(stopwords_set)
        logger.info("读取停用词 stopwords 共{0} 个from {1}".format(len(stopwords_ls), path))
        return stopwords_ls


def read_yaml(content: Text, reader_type: Union[Text, List[Text]] = "safe") -> Any:
    """Parses yaml from a text.

    Args:
        content: A text containing yaml content.
        reader_type: Reader type to use. By default "safe" will be used.

    Raises:
        ruamel.yaml.parser.ParserError: If there was an error when parsing the YAML.
    """
    if _is_ascii(content):
        # Required to make sure emojis are correctly parsed
        content = (
            content.encode("utf-8")
                .decode("raw_unicode_escape")
                .encode("utf-16", "surrogatepass")
                .decode("utf-16")
        )

    yaml_parser = yaml.YAML(typ=reader_type)
    # yaml = YAML(typ='safe', pure=True)
    yaml_parser.version = YAML_VERSION
    yaml_parser.preserve_quotes = True

    return yaml_parser.load(content) or {}


def _is_ascii(text: Text) -> bool:
    return all(ord(character) < 128 for character in text)


def read_yaml_file(filename: Union[Text, Path]) -> Union[List[Any], Dict[Text, Any]]:
    """Parses a yaml file.

    Raises an exception if the content of the file can not be parsed as YAML.

    Args:
        filename: The path to the file which should be read.

    Returns:
        Parsed content of the file.
    """
    # from ruamel import yaml
    # with open(yaml_path, 'r') as f:
    #     content = f.read()
    # # 通过load函数将数据转化为列表或字典
    # data_yaml = yaml.load(content,Loader=yaml.RoundTripLoader)
    # 返回对象 ： ruamel.yaml.comments.CommentedMap

    try:
        return read_yaml(read_file(filename, DEFAULT_ENCODING))
    except (YAMLError, DuplicateKeyError) as e:
        raise YamlSyntaxException(filename, e)


def write_yaml(
        data: Any,
        target: Union[Text, Path, StringIO],
        should_preserve_key_order: bool = False,
) -> None:
    """Writes a yaml to the file or to the stream

    Args:
        data: The data to write.
        target: The path to the file which should be written or a stream object
        should_preserve_key_order: Whether to force preserve key order in `data`.
    """
    _enable_ordered_dict_yaml_dumping()

    if should_preserve_key_order:
        data = convert_to_ordered_dict(data)

    dumper = yaml.YAML()
    # no wrap lines
    dumper.width = YAML_LINE_MAX_WIDTH

    # use `null` to represent `None`
    dumper.representer.add_representer(
        type(None),
        lambda self, _: self.represent_scalar("tag:yaml.org,2002:null", "null"),
    )

    if isinstance(target, StringIO):
        dumper.dump(data, target)
        return

    with Path(target).open("w", encoding=DEFAULT_ENCODING) as outfile:
        dumper.dump(data, outfile)


def _enable_ordered_dict_yaml_dumping() -> None:
    """Ensure that `OrderedDict`s are dumped so that the order of keys is respected."""
    yaml.add_representer(
        OrderedDict,
        RoundTripRepresenter.represent_dict,
        representer=RoundTripRepresenter,
    )


def convert_to_ordered_dict(obj: Any) -> Any:
    """Convert object to an `OrderedDict`.

    Args:
        obj: Object to convert.

    Returns:
        An `OrderedDict` with all nested dictionaries converted if `obj` is a
        dictionary, otherwise the object itself.
    """
    if isinstance(obj, OrderedDict):
        return obj
    # use recursion on lists
    if isinstance(obj, list):
        return [convert_to_ordered_dict(element) for element in obj]

    if isinstance(obj, dict):
        out = OrderedDict()
        # use recursion on dictionaries
        for k, v in obj.items():
            out[k] = convert_to_ordered_dict(v)

        return out

    # return all other objects
    return obj


def write_to_yaml(data: Union[List, Dict] = None, path: Path = None):
    """
    将 python object 数据转换为 yaml 数据
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # def represent_str(dumper, instance):
    #     if "\n" in instance:
    #         return dumper.represent_scalar('tag:yaml.org,2002:str',
    #                                        instance,
    #                                        style='|')
    #     else:
    #         return dumper.represent_scalar('tag:yaml.org,2002:str',
    #                                        instance)
    # yaml.add_representer(str,represent_str)

    # 转换为 python 对象
    with open(path, 'w', encoding='utf-8') as f:
        # Yaml format
        # yaml.dump()方法不会将列表或字典数据进行转化yaml标准模式，只会将数据生成到yaml文档中
        # 使用ruamel模块中的yaml方法生成标准的yaml文档 indent=2,
        yaml.dump(data, f,
                  encoding='utf-8',
                  default_flow_style=False,
                  allow_unicode=True,
                  Dumper=yaml.RoundTripDumper)


def dataframe_to_file(path: Path, data: pd.DataFrame = None, mode: Text = 'w', format=None,
                      index=True, index_col=None, sheet_name='Sheet1', over_write_sheet=True,
                      orient='records', dtype=None, lines=True, encoding='utf-8'):
    if format is None:
        path_dir, filename = os.path.split(path)
        filename, suffix = os.path.splitext(filename)
        format = suffix[1:]
    if mode == 'w':
        os.makedirs(os.path.dirname(path), exist_ok=True)
    elif mode == 'r' and not os.path.exists(path):
        return

    if format == 'csv':
        if mode == 'w':
            data.to_csv(path_or_buf=path, encoding='utf_8_sig', quoting=1, index=False)
        elif mode == 'r':
            data = pd.read_csv(filepath_or_buffer=path, encoding='utf_8_sig', quoting=1, index_col=None)
    elif format == 'json':
        if mode == 'w':
            # intent 参数加入后，pd.read_json会报错
            data.to_json(path_or_buf=path, orient=orient, force_ascii=False, lines=lines)
        elif mode == 'r':
            data = pd.read_json(path_or_buf=path, orient=orient, dtype=dtype, lines=lines, encoding=encoding)

    elif format == 'xlsx' or format == 'xls':
        if mode in ['w', 'a']:
            # data.to_excel(path,index=index )
            dataframe_to_excel(excel_path=path, df=data, mode=mode, sheet_name=sheet_name,
                               index=index,
                               over_write_sheet=over_write_sheet)
        elif mode == 'r':
            data = pd.read_excel(path, index_col=index_col, sheet_name=sheet_name, dtype=dtype)
    elif format == 'yml':
        if mode == 'w':
            write_to_yaml(path=path, data=data.to_dict(orient='records'))
        if mode == 'r':
            data_json = read_yaml_file(path=path)
            data = pd.DataFrame(data=data_json)

    # 去除重复数据
    # data.drop_duplicates(inplace=True)
    # 去除字符串数据的两端空格
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return data


def dataframe_to_excel(excel_path: Path, df: pd.DataFrame, mode='a',
                       sheet_name: Text = 'Sheet1', index=False, over_write_sheet=True):
    if mode == 'a':
        engine = "openpyxl"
    else:
        engine = None
    if not os.path.exists(excel_path):
        mode = 'w'
    writer = pd.ExcelWriter(excel_path, mode=mode, engine=engine)

    wb = writer.book
    if over_write_sheet and sheet_name in wb.sheetnames:
        wb.remove(wb[sheet_name])

    df.to_excel(writer, sheet_name, index=index)
    writer.save()
    writer.close()


def is_sheet_exist(excel_path, sheet_name):
    sheet_exist = False
    if os.path.exists(excel_path):
        writer = pd.ExcelWriter(excel_path, mode="a", engine="openpyxl")
        # pandas操作Excel底层也是依赖于其它的模块, 比如xlrd、openpyxl
        # 所以这里的 wb = writer.book  就相当于  from openpyxl import load_workbook; wb = load_workbook("xxx.xlsx")
        wb = writer.book
        # 查看已存在的所有的sheet, 总共是5个, "b1"和"b2"是自动创建的, 因为"b"已经存在了, 我们又导入了两次
        if sheet_name in wb.sheetnames:
            sheet_exist = True
        writer.close()
    return sheet_exist


def get_matched_filenames(data_dir=None, filename_pattern=None):
    try:
        if os.path.exists(data_dir):
            re_compiled = re.compile(pattern=filename_pattern)
            matched_filenames = [f for f in os.listdir(data_dir) if re_compiled.match(f)]
            return matched_filenames
        else:
            raise IOError(f"data_dir= {data_dir} not exists")
    except Exception as e:
        raise IOError(f"data_dir= {data_dir} is not a directory:{e}")


def get_file_dir(file, directory: Text = None, filename: Text = None):
    file_dir = os.path.dirname(file)
    if directory:
        file_dir = os.path.join(file_dir, directory)
    if filename:
        file_dir = os.path.join(file_dir, filename)
    file_dir = os.path.normpath(file_dir)
    return file_dir


def get_file_stem(file, full_name=False):
    if full_name:
        return Path(file).name
    else:
        return Path(file).stem


def _to_yaml_examples_dict(key, value, examples, examples_key='examples', ):
    """
    将 列表形式的 examples 转换为 yaml 格式
    """
    from ruamel.yaml.scalarstring import PreservedScalarString as PSS
    #  对 examples 进行排序： 文本长度+首字符
    examples = sorted(examples, key=lambda x: (len(x), x.lower()))
    #  使用  PreservedScalarString: https://mlog.club/article/1462558
    examples_str = "".join([f"- {example}\n" for example in examples if example.strip() != ''])
    yaml_examples_dict = {key: value, examples_key: PSS(examples_str)}
    return yaml_examples_dict


def _to_yaml_string_dict(key_value_dict: Dict[Text, Text], examples, examples_key='examples', ):
    """
    将 列表形式的 examples 转换为 yaml 格式
    key_value_dict: 可以包含多个key
    """
    from ruamel.yaml.scalarstring import PreservedScalarString as PSS
    #  对 examples 进行排序： 文本长度+首字符
    examples = sorted(examples, key=lambda x: (len(x), x.lower()))
    #  使用  PreservedScalarString: https://mlog.club/article/1462558
    examples_str = "".join([f"- {example}\n" for example in examples if example.strip() != ''])
    key_value_dict.update({examples_key: PSS(examples_str)})
    return key_value_dict


def get_files_from_dir(dir, return_filename=False) -> List[Text]:
    if os.path.isdir(dir):
        filenames = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    if return_filename:
        return filenames
    else:
        filepath = [os.path.join(dir, f) for f in filenames]
        return filepath


def set_working_dir(project_name='faq'):
    if project_name:
        project_dir = Path.home().joinpath(project_name)
    else:
        script_path = os.path.abspath(__file__)
        project_dir = Path(script_path).parent.parent.absolute()
    os.chdir(project_dir)
    # sys.path.extend([str(project_dir)])
    sys.path.insert(1, project_dir)
    print(f"project_dir={project_dir}")
    # print(f"sys.path={sys.path}")
    return project_dir


def create_temporary_directory() -> Text:
    """Creates a tempfile.TemporaryDirectory."""
    f = tempfile.TemporaryDirectory()
    return f.name


def create_directory(directory_path: Text) -> None:
    """Creates a directory and its super paths.

    Succeeds even if the path already exists."""

    try:
        os.makedirs(directory_path)
    except OSError as e:
        # be happy if someone already created the path
        if e.errno != errno.EEXIST:
            raise


def create_directory_for_file(file_path: Union[Text, Path]) -> None:
    """Creates any missing parent directories of this file path."""

    create_directory(os.path.dirname(file_path))


def pickle_dump(filename: Union[Text, Path], obj: Any) -> None:
    """Saves object to file.

    Args:
        filename: the filename to save the object to
        obj: the object to store
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(filename: Union[Text, Path]) -> Any:
    """Loads an object from a file.

    Args:
        filename: the filename to load the object from

    Returns: the loaded object
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


def unarchive(byte_array: bytes, directory: Text) -> Text:
    """Tries to unpack a byte array interpreting it as an archive.

    Tries to use tar first to unpack, if that fails, zip will be used."""

    try:
        tar = tarfile.open(fileobj=IOReader(byte_array))
        tar.extractall(directory)
        tar.close()
        return directory
    except tarfile.TarError:
        zip_ref = zipfile.ZipFile(IOReader(byte_array))
        zip_ref.extractall(directory)
        zip_ref.close()
        return directory



def read_config_file(filename: Union[Path, Text]) -> Dict[Text, Any]:
    """Parses a yaml configuration file. Content needs to be a dictionary.

    Args:
        filename: The path to the file which should be read.

    Raises:
        YamlValidationException: In case file content is not a `Dict`.

    Returns:
        Parsed config file.
    """
    from rasa.shared.constants import CONFIG_SCHEMA_FILE
    return read_validated_yaml(filename, CONFIG_SCHEMA_FILE)


def read_validated_yaml(filename: Union[Text, Path], schema: Text) -> Any:
    """Validates YAML file content and returns parsed content.

    Args:
        filename: The path to the file which should be read.
        schema: The path to the schema file which should be used for validating the
            file content.

    Returns:
        The parsed file content.

    Raises:
        YamlValidationException: In case the model configuration doesn't match the
            expected schema.
    """
    content = read_file(filename)

    # from rasa.shared.utils.validation import validate_yaml_schema
    from utils.validation import validate_yaml_schema
    validate_yaml_schema(content, schema)
    return read_yaml(content)


