import inspect
import sys
import numpy as np
import math


'''
Raise a "not defined" exception as a reminder 
'''
def _raise_not_defined():
    print "Method not implemented: %s" % inspect.stack()[1][3]
    sys.exit(1)


'''
Extract 'basic' features, i.e., whether a pixel is background or
forground (part of the digit) 
'''
def extract_basic_features(digit_data, width, height):
    features=[]
    # Your code starts here
    
    for y in range(height):
        for x in range(width):
            if digit_data[y][x] == 0:
                features.append(0)
            else:
                features.append(1)
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here
    return features

'''
Extract advanced features that you will come up with 
'''
#features1: count only # not +
def features1(digit_data, width, height):
    features = []
    for y in range(height):
        for x in range(width):
            if 2 in digit_data[y] and 1 not in digit_data[y]:
                if digit_data[x][y] == 2:
                    features.append(1)
                else:
                    features.append(0)
            else:
                if digit_data[y][x] == 1:
                    features.append(1)
                else:
                    features.append(0)
    return features
#features2: remove noises
def features2(digit_data,width,height):
    features = []
    for y in range(height):
        for x in range(width):
            if x == 0 and digit_data[y][x+1] == 0 and digit_data[y][x] != 0:
                features.append(0)
            elif x == width - 1 and digit_data [y][x-1] == 0 and digit_data[y][x] != 0:
                features.append(0)
            elif digit_data[y][x-1] == 0 and digit_data [y][x-1] == 0 and digit_data[y][x] != 0:
                features.append(0)
            elif digit_data[y][x] == 0:
                features.append(0)
            else:
                features.append(1)
    return features
#features3: instead of binary, count 2 as well
def features3(digit_data,width,height):
    features = []
    for y in range(height):
        for x in range(width):
            features.append(digit_data[y][x]/1.3)
    return features

def extract_advanced_features(digit_data, width, height):
    #change this to test
    # Your code starts here
    #features1: count only # not +
    features = []
    features_basic = extract_basic_features(digit_data,width,height)
    features_one = features1(digit_data,width,height)
    features_two = features2(digit_data,width,height)
    features_three = features3(digit_data,width,height)
    
    for x in range((width*height)):
        val1 = float(features_basic[x])
        val2 = float(features_one[x])
        val3 = float(features_two[x])
        val4 = float(features_three[x])
        features.append((val2 + val3 + val4) / 3)
#    features_three = features3(digit_data,width,height)

    
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here
    return features

'''
Extract the final features that you would like to use
'''
def extract_final_features(digit_data, width, height):
   return extract_basic_features(digit_data,width,height)

'''
Compupte the parameters including the prior and and all the P(x_i|y). Note
that the features to be used must be computed using the passed in method
feature_extractor, which takes in a single digit data along with the width
and height of the image. For example, the method extract_basic_features
defined above is a function than be passed in as a feature_extractor
implementation.

The percentage parameter controls what percentage of the example data
should be used for training. 
'''
def compute_statistics(data, label, width, height, feature_extractor, percentage=90.0):
    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here
    global prior
    global conditional
    percentage_given = percentage * 0.01
    used_data_count = len(label) * percentage_given
    #prior probability
    prior = np.zeros((10,),dtype = np.float)
    for x in range(int(used_data_count)):
        tmp = int(label[x])
        prior[tmp] += 1
    prior /= used_data_count
    

    #conditional probability
    #get all the features from the used data
    total_feature_data = []
    for x in range(int(used_data_count)):
        feature = feature_extractor(data[x],width,height)
        total_feature_data.append(feature)


    #(number of true in pixel for specific label + laplaceK) / (total number of pixels + total number of 1 in specific pixel + laplaceK)
    s = (10, width * height)
    tmp = np.zeros(s)
    for i in range(int(used_data_count)):
        label_number = label[i]
        for x in range(len(total_feature_data[i])):
            tmp[label_number][x] += total_feature_data[i][x]

    #laplaceK: Can be changed
    laplaceK = 0.0001
    nominator_each_pixel = np.zeros(s)
    denominator_each_pixel = np.zeros(s)

    #nominator

    for i in range(len(tmp)):
        for j in range(len(tmp[i])):
            nominator_each_pixel[i][j] = tmp[i][j] + laplaceK

    #denominator
    for i in range(len(tmp)):
        for j in range(len(tmp[i])):
            denominator_each_pixel[i][j] = (prior[i] * used_data_count) + laplaceK
    conditional = np.zeros(s)
    #conditional probability each pixel
    for i in range(len(tmp)):
        for j in range(len(tmp[i])):
            conditional[i][j] = nominator_each_pixel[i][j]/denominator_each_pixel[i][j]

'''
For the given features for a single digit image, compute the class 
'''
def compute_class(features):
    predicted = -1
    prediction_prob = []
    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    for i in range(len(conditional)):
        tmp = 1
        count = 0
        for j in range(len(conditional[i])):
            if features[count] == 1:
                tmp += math.log(conditional[i][j] * 3)
            elif features[count] > 0:
                tmp += math.log(conditional[i][j])
            else:
                tmp += math.log(1 - conditional[i][j])
            count += 1
        
        tmp += math.log(prior[i])
        tmp = math.exp(tmp)
        prediction_prob.append(tmp)
    predicted = np.argmax(prediction_prob)

    # Your code ends here
    return predicted

'''
Compute joint probaility for all the classes and make predictions for a list
of data
'''
def classify(data, width, height, feature_extractor):
    predicted=[]
    for i in range(len(data)):
        features = feature_extractor(data[i],width,height)
        predict_image = compute_class(features)

        predicted.append(predict_image)
    

    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here

    return predicted







        
    
