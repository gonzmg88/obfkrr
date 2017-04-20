#   --------------------------------------
#   Copyright & Disclaimer
#   --------------------------------------
#
#   The programs contained in this package are granted free of charge for
#   research and education purposes only. Scientific results produced using
#   the software provided shall acknowledge the use of this implementation
#   provided by us. If you plan to use it for non-scientific purposes,
#   don't hesitate to contact us. Because the programs are licensed free of
#   charge, there is no warranty for the program, to the extent permitted
#   by applicable law. except when otherwise stated in writing the
#   copyright holders and/or other parties provide the program "as is"
#   without warranty of any kind, either expressed or implied, including,
#   but not limited to, the implied warranties of merchantability and
#   fitness for a particular purpose. the entire risk as to the quality and
#   performance of the program is with you. should the program prove
#   defective, you assume the cost of all necessary servicing, repair or
#   correction. In no event unless required by applicable law or agreed to
#   in writing will any copyright holder, or any other party who may modify
#   and/or redistribute the program, be liable to you for damages,
#   including any general, special, incidental or consequential damages
#   arising out of the use or inability to use the program (including but
#   not limited to loss of data or data being rendered inaccurate or losses
#   sustained by you or third parties or a failure of the program to
#   operate with any other programs), even if such holder or other party
#   has been advised of the possibility of such damages.
#
#   NOTE: This is just a demo providing a default initialization. Training
#   is not at all optimized. Other initializations, optimization techniques,
#   and training strategies may be of course better suited to achieve improved
#   results in this or other problems.
#
# Copyright (c) 2017 by Gonzalo Mateo-Garcia
# gonzalo.mateo-garcia@uv.es
# http://isp.uv.es/
#
import logging
import sys

def empty(logger):
    for hd in logger.handlers:
        logger.removeHandler(hd)
    return logger

def screen_logger(logger,empty_logger=True):
    if empty_logger:
        empty(logger)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
       '%(asctime)-12s %(name)-12s %(levelname)-8s %(message)s',
       datefmt="%Y-%m-%d %H:%M:%S")
    
    # logger to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # logger to stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    handler.setLevel(logging.WARN)
    logger.addHandler(handler)
    return logger

def file_logger(nombre,logger,empty_logger=True):
    if empty_logger:
        empty(logger)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
       '%(asctime)-12s %(name)-12s %(levelname)-8s %(message)s',
       datefmt="%Y-%m-%d %H:%M:%S")
    
    
    # logger to stdout
    handler = logging.FileHandler(nombre+".log")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # logger to stderr
    handler = logging.FileHandler(nombre+".error")
    handler.setFormatter(formatter)
    handler.setLevel(logging.WARN)
    logger.addHandler(handler)
    return logger

    