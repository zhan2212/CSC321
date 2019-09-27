# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 19:00:37 2018

@author: zyy19
"""

import language_model as lm 
import checking as ck 

if __name__ == "__main__":
    
    # train model
    model = lm.train(16,128)
    
    
    # tsne 
    model.tsne_plot()
    
    
    
    print(model.word_distance("new","york"))
    model.display_nearest_words("new")
    model.display_nearest_words("york")
    # 4
    print(model.word_distance("government","political"))
    print(model.word_distance("government","university"))
    
    
    