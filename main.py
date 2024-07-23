import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Tuple, Union
import numpy as np
from sklearn.metrics import accuracy_score
import random