import os
import pyarrow as pa
from abc import ABCMeta, abstractmethod
from pyarrow import parquet as pq
# from transformers import AutoTokenizer
from tokenizers import Tokenizer
from typing import List, Set


"""
FileHandlers implement basic file operations such as file type checking, opening, indexing, and slicing. 
By implementing these basic operations, users can add support for arbitrary file types to ScalableReader.
"""


class ShardFileHandler(object, metaclass=ABCMeta):
    """
    Stub for shard file readers of different formats.
    Must implement open, length, indexing, and slicing functions.
    """

    def is_legal(self, filepath: str):
        """
        Given a file path, determine if it qualifies for this handler.
        Ideally does not involve opening the file.
        """
        return os.path.isfile(filepath)

    @abstractmethod
    def open(self, path: str):
        """
        Open the file, to be indexed via self.get() method.
        Avoid reading entire multi-Gb files when possible!
        """
        pass

    @abstractmethod
    def length(self, path: str):
        """
        Calculate the number of documents in the given file.
        Avoid reading entire multi-Gb files when possible!
        """
        pass

    @abstractmethod
    def get(self, reader, index: int, drop_tokens: Set):
        """
        Given the output of self.open() and an index, return the document at that index.
        Then, remove the first and/or last items if they appear in drop_tokens.
        Try to avoid reading entire documents at a time in case of long documents,
        but this is less important than avoiding reading entire files as above.
        Output must support len() method.
        """
        pass

    @abstractmethod
    def slice(self, doc, index: int, n_pull: int) -> List:
        """
        Given a long document, retrieve n_pull consecutive items starting from index.
        Again, try to be memory-efficient when doing so, but efficiency in self.get()
        and self.open() is far more important.
        Must return a python list.
        """
        pass


class ArrowHandler(ShardFileHandler):
    """
    Reader for indexable, pre-tokenized PyArrow shard files.
    Pyarrow shard files are expected to hold multiple RecordBatches,
    where each RecordBatch has a "tokens" field consisting of
    a single token list (i.e. each document is a single sequence
    under a "token" field, and the file is a list of such sequences).

    A preferred format as we can load document chunks without having to ever pull
    the entire document or shard file, allowing for graceful handling of large documents.
    Non-standard data format, though.
    """

    def __init__(self, col_names: List[str] = ["text", "contents", "tokens"]):
        self.col_names = col_names

    def is_legal(self, filepath: str):
        return "arrow" in os.path.splitext(filepath)[1]

    def open(self, path: str):
        return pa.ipc.open_file(pa.memory_map(path))

    def length(self, path: str):
        return self.open(path).num_record_batches

    def get(self, reader: pa.RecordBatchFileReader, index: int, drop_tokens: Set):
        assert (
            index < reader.num_record_batches
        ), f"Illegal index {index} in set of {reader.num_record_batches} documents"
        frame = reader.get_batch(index)
        doc = None
        for name in self.col_names:
            if name in frame.column_names:
                doc = frame[name]
                break
        assert (
            doc is not None
        ), f"None of column names {self.col_names} found in file headers {frame.column_names}"
        if len(doc) > 0 and doc[0].as_py() in drop_tokens:
            doc = doc.slice(1, len(doc) - 1)
        # Recheck len for edge case where doc=[eos]
        if len(doc) > 0 and doc[-1].as_py() in drop_tokens:
            doc = doc.slice(0, len(doc) - 1)
        return doc

    def slice(self, doc: pa.UInt32Array, index: int, n_pull: int) -> List:
        return doc.slice(index, n_pull).to_pylist()


class ParquetHandler(ShardFileHandler):
    """
    Reader for indexable parquet shard files, common in HF datasets.
    Here we assume reasonably small shard files (<5Gb) and truncate docs to max_doclen characters,
    as we rely on parquet/pandas for efficient file reading, and tokenize entire documents
    before getting/slicing. However, this is a standard and widely-used data format.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        col_names: List[str] = ["text", "contents", "tokens"],
        max_doclen: int = 1_000_000,
    ):
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer = tokenizer
        self.col_names = col_names
        self.max_doclen = max_doclen

    def is_legal(self, filepath: str):
        return "parquet" in os.path.splitext(filepath)[1]

    def open(self, path: str):
        names = pq.read_schema(path).names
        match = None
        for name in self.col_names:
            if name in names:
                match = name
                break
        assert (
            match is not None
        ), f"None of column names {self.col_names} found in file headers {names}"
        return pq.read_pandas(path, columns=[match], partitioning=None)[match]

    def length(self, path: str):
        try:
            return pq.read_metadata(path).num_rows
        except:
            print("Offending path:", path)

    def get(self, reader, index: int, drop_tokens: Set):
        assert (
            index < reader.length()
        ), f"Illegal index {index} in set of {reader.length()} documents"
        doc = self.tokenizer.encode(str(reader[index])[: self.max_doclen])
        if len(doc) > 0 and doc[0] in drop_tokens:
            doc = doc[1:]
        # Recheck len for edge case where doc=[eos]
        if len(doc) > 0 and doc[-1] in drop_tokens:
            doc = doc[:-1]
        return doc

    def slice(self, doc: List, index: int, n_pull: int) -> List:
        return doc[index : index + n_pull]