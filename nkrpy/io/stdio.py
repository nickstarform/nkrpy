# flake8: noqa
# TODO: work on this
__filename__ = __file__.split('/')[-1].strip('.py')

class File(object):
  """ An helper class for file reading 
      EXAMPLE:

      with File('file.name') as f:
         print f.head(5)
         print f.tail(5)
         for row in f.backward():
             print row """

  def __init__(self, *args, **kwargs):
    super(File, self).__init__(*args, **kwargs)
    self.BLOCKSIZE = 4096

  def head(self, lines_2find=1):
    self.seek(0)              #Rewind file
    return [super(File, self).next() for x in xrange(lines_2find)]

  def tail(self, lines_2find=1):  
    self.seek(0, 2)             #Go to end of file
    bytes_in_file = self.tell()
    lines_found, total_bytes_scanned = 0, 0
    while (lines_2find + 1 > lines_found and
         bytes_in_file > total_bytes_scanned): 
      byte_block = min(
        self.BLOCKSIZE,
        bytes_in_file - total_bytes_scanned)
      self.seek( -(byte_block + total_bytes_scanned), 2)
      total_bytes_scanned += byte_block
      lines_found += self.read(self.BLOCKSIZE).count('\n')
    self.seek(-total_bytes_scanned, 2)
    line_list = list(self.readlines())
    return line_list[-lines_2find:]

  def backward(self):
    self.seek(0, 2)             #Go to end of file
    blocksize = self.BLOCKSIZE
    last_row = ''
    while self.tell() != 0:
      try:
        self.seek(-blocksize, 1)
      except IOError:
        blocksize = self.tell()
        self.seek(-blocksize, 1)
      block = self.read(blocksize)
      self.seek(-blocksize, 1)
      rows = block.split('\n')
      rows[-1] = rows[-1] + last_row
      while rows:
        last_row = rows.pop(-1)
        if rows and last_row:
          yield last_row
    yield last_row

import os
import numpy as np
import gc
import multiprocessing as mp


def chunkify_file(fname, size=1024*1024*1000, skiplines=-1):
    """
    function to divide a large text file into chunks each having size ~= size so that the chunks are line aligned

    Params : 
        fname : path to the file to be chunked
        size : size of each chink is ~> this
        skiplines : number of lines in the begining to skip, -1 means don't skip any lines
    Returns : 
        start and end position of chunks in Bytes
    """
    chunks = []
    fileEnd = os.path.getsize(fname)
    with open(fname, "rb") as f:
        if(skiplines > 0):
            for i in range(skiplines):
                f.readline()

        chunkEnd = f.tell()
        count = 0
        while True:
            chunkStart = chunkEnd
            f.seek(f.tell() + size, os.SEEK_SET)
            f.readline()  # make this chunk line aligned
            chunkEnd = f.tell()
            chunks.append((chunkStart, chunkEnd - chunkStart, fname))
            count+=1

            if chunkEnd > fileEnd:
                break
    return chunks

def parallel_apply_line_by_line_chunk(chunk_data):
    """
    function to apply a function to each line in a chunk

    Params :
        chunk_data : the data for this chunk 
    Returns :
        list of the non-None results for this chunk
    """
    chunk_start, chunk_size, file_path, func_apply = chunk_data[:4]
    func_args = chunk_data[4:]

    t1 = time.time()
    chunk_res = []
    with open(file_path, "rb") as f:
        f.seek(chunk_start)
        cont = f.read(chunk_size).decode(encoding='utf-8')
        lines = cont.splitlines()

        for i,line in enumerate(lines):
            ret = func_apply(line, *func_args)
            if(ret != None):
                chunk_res.append(ret)
    return chunk_res

def parallel_apply_line_by_line(input_file_path, chunk_size_factor, num_procs, skiplines, func_apply, func_args, fout=None):
    """
    function to apply a supplied function line by line in parallel

    Params :
        input_file_path : path to input file
        chunk_size_factor : size of 1 chunk in MB
        num_procs : number of parallel processes to spawn, max used is num of available cores - 1
        skiplines : number of top lines to skip while processing
        func_apply : a function which expects a line and outputs None for lines we don't want processed
        func_args : arguments to function func_apply
        fout : do we want to output the processed lines to a file
    Returns :
        list of the non-None results obtained be processing each line
    """
    num_parallel = min(num_procs, mp.cpu_count()) - 1

    jobs = chunkify_file(input_file_path, 1024 * 1024 * chunk_size_factor, skiplines)

    jobs = [list(x) + [func_apply] + func_args for x in jobs]

    print("Starting the parallel pool for {} jobs ".format(len(jobs)))

    lines_counter = 0

    pool = mp.Pool(num_parallel, maxtasksperchild=1000)  # maxtaskperchild - if not supplied some weird happend and memory blows as the processes keep on lingering

    outputs = []
    for i in range(0, len(jobs), num_parallel):
        print("Chunk start = ", i)
        t1 = time.time()
        chunk_outputs = pool.map(parallel_apply_line_by_line_chunk, jobs[i : i + num_parallel])

        for i, subl in enumerate(chunk_outputs):
            for x in subl:
                if(fout != None):
                    print(x, file=fout)
                else:
                    outputs.append(x)
                lines_counter += 1
        del(chunk_outputs)
        gc.collect()
        print("All Done in time ", time.time() - t1)

    print("Total lines we have = {}".format(lines_counter))

    pool.close()
    pool.terminate()
    return outputs
