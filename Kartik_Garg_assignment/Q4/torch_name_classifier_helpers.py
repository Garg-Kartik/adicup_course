# import some Python 3 utilities
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

import numpy as np

# for opening a file
from io import open

# string manipulation utilities
import unicodedata
import string

# random numbers
import random
      
# a helper function to convert a unicode string
# to plain old ascii.
# NFD - Normal Form D (i.e. cannonical)
#  Mn - Mark, Non-Spacing
def unicodeToASCII(s, letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in letters
        )

# Read a file and split into lines, taking care
# to convert to ASCII as well.
def readLines(filename, letters):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToASCII(line, letters) for line in lines]

# helper function to convert softmax output to corresponding most 
# probable category and assocaited category index
def categoryFromOutput(output, categories):
    topk_val, top_idx = output.topk(1)
    category_idx = top_idx[0].item()
    return categories[category_idx], category_idx

# helper function to convert a one-hot tensor to text
def textFromTensor(one_hot_tensor, letters):
    the_text = ""
    ltr_val, ltr_idx = one_hot_tensor.topk(1)
    for i in range(0, one_hot_tensor.size(0)):
        the_text += letters[ltr_idx[i].item()]        
    return the_text

# a helper function to convert a category to a one-hot tensor
def categoryToTensor(the_category, categories):
    cat_idx = categories.index(the_category)
    one_hot_tensor = torch.zeros(1, len(categories))
    one_hot_tensor[0][cat_idx] = 1
    return one_hot_tensor

# helper function to convert text to a one-hot tensor
def textToTensor(the_text, letters):
    one_hot_tensor = torch.zeros(len(the_text), 1, len(letters))
    for idx, letter in enumerate(the_text):
        ltr_idx = letters.find(letter)
        one_hot_tensor[idx][0][ltr_idx] = 1
    return one_hot_tensor

# helper function to convert text to a long tensor of targets
# (i.e. letter indices of next chars in sequence)
def textToTargetTensor(the_text, letters):
    letter_indexes = [letters.find(the_text[idx]) for idx in range(1, len(the_text))]
    letter_indexes.append(len(letters) - 1) # EOS
    return torch.LongTensor(letter_indexes)

# a helper function to select a random entry from the
# given list
def randomChoice(the_list):
    return the_list[random.randint(0, len(the_list) - 1)]

# a helper function that generates a random category and
# random line from that category
def randomTrainingPair(categories, lines):
    category = randomChoice(categories)
    line = randomChoice(lines[category])
    return category, line

# a helper function that constructs feature and label tensors 
# for a randomly chosen category-line (language-name) pair
def randomTrainingExample(categories, lines, letters):
    category = randomChoice(categories)
    cat_idx = categories.index(category)
    line = randomChoice(lines[category])
    category_tensor = torch.tensor([cat_idx], dtype=torch.long)
    line_tensor = textToTensor(line, letters)
    return category, line, category_tensor, line_tensor

