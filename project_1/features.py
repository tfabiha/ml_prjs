# obtains the number of words with all characters in upper case feature
# takes as input all the tweets from our data file to observe all the caps words per tweet
def get_caps_encoding(tweets):
    # all caps
    tot_caps = []
    for count, tweet in enumerate(tweets):
        temp_counter = 0
        tweet = tweet.split(' ')
        
        for word in tweet:
            if word.isupper():
                temp_counter += 1
        tot_caps.append(temp_counter)
    return tot_caps

# obtains the number of occurrences of each part-of-speech tag feature
# takes as input all the pos tags from our data file to observe all the pos features per tweet
def get_pos_encoding(POS_tags):
    # POS tags
    tags = "NOS^ZLMVAR!DP&TXY#@~UE$,G"
    tot_tags = []
    for count, tweet in enumerate(POS_tags):
        temp_counter = [0 for each in tags]
        for t in tweet:
            if tags.find(t) > -1:
                temp_counter[tags.find(t)] += 1
        tot_tags.append(temp_counter)
    return tot_tags

# combines the POS and the caps features into one
def get_encoding_features(tweets, POS_tags):
    tot_caps = get_caps_encoding(tweets)
    tot_tags = get_pos_encoding(POS_tags)
    
    tot = []
    for i in range(len(tot_tags)):
        tot.append(tot_tags[i] + [tot_caps[i]])
    return tot

# obtains the 16 lexicon features against Hashtag-Sentiment
def get_lexHS_features(tweets, lex_dir):
    return get_lex_features(tweets, lex_dir + "Hashtag-Sentiment-Lexicon/HS-unigrams.txt", lex_dir + "Hashtag-Sentiment-Lexicon/HS-bigrams.txt")

# obtains the 16 lexicon features against Sentiment140
def get_lexSent_features(tweets, lex_dir):
    return get_lex_features(tweets, lex_dir + "Sentiment140-Lexicon/Emoticon-unigrams.txt", lex_dir + "Sentiment140-Lexicon/Emoticon-bigrams.txt")
    
# given the uni and bi lexicon files obtain the 16 lex features
def get_lex_features(tweets, uni, bi):
    from nltk import bigrams
    
    f = []
    f.append(open(uni, "r") )
    f.append(open(bi, "r") )
    
    lex_features = []
    lex_features.append([])
    lex_features.append([])
    
    # we are running this loop twice, once for unigram lex and once for bigram lex
    for c, file in enumerate(f):
        lex = {}
        for count, line in enumerate(file):
            line = line.strip('\n')
            line = line.split('\t')
        
            lex[line[0]] = float(line[1])
        file.close()
    
        # for every tweet find the 8 features listed below
        for count, tweet in enumerate(tweets):
            count_pos_tokens = 0
            count_neg_tokens = 0
            sum_pos_score = 0
            sum_neg_score = 0
            max_pos_score = 0
            max_neg_score = 0
            last_pos_score = 0
            last_neg_score = 0
            
            if c == 0: # working with unigrams
                tweet = tweet.split(' ')
            else: # working with bigrams
                tweet = list(bigrams(tweet.split(" ")))
                tweet = list(map(' '.join, tweet))

            for word in tweet:
                if word in lex:
                    score = lex[word]

                    if score > 0: # positive polarity
                        count_pos_tokens += 1
                        last_pos_score = score
                        sum_pos_score += score

                        if count == 0:
                            max_pos_score = score
                        elif max_pos_score < score:
                            max_pos_score = score
                    elif score < 0: # negative polarity
                        count_neg_tokens += 1
                        last_neg_score = -1 * score
                        sum_neg_score += -1 * score

                        if count == 0:
                            max_neg_score = -1 * score
                        elif max_neg_score <  -1 * score:
                            max_neg_score = -1 * score
                
        
            lex_features[c].append([ count_pos_tokens, count_neg_tokens, sum_pos_score, sum_neg_score, max_pos_score, max_neg_score, last_pos_score, last_neg_score ])
    
    l = []
    for i in range(len(lex_features[0])):
        l.append(lex_features[0][i] + lex_features[1][i])
    return l
    
# opens the data file and extracts the tokens, pos_tags and labels
# cleans the data somewhat to make it easy for parsing
def extract(file_name):

    ext = {}
    ext["tokens"] = []
    ext["pos_tags"] = []
    ext["label"] = []
    
    f = open(file_name, "r")
    for count, line in enumerate(f):
        if count != 0:
            line = line.strip('\n')
            #print(line)

            # first_comma and last_comma are the points at which we split the dataset
            last_comma = line.rfind(',')
            first_comma = comma_outta_quote(line)

            token = line[: first_comma].strip("\"")

            pos_tags = line[first_comma + 1 : last_comma].strip("\"")
            pos_tags = pos_tags.split(" ")

            ext["tokens"].append( token ) # an array of strings, each string represents a tweet
            ext["pos_tags"].append( pos_tags ) # an array of arrays, each row contains an array of all the POS tags
            ext["label"].append( line[last_comma + 1 :] ) # an array of labels: negative, positive, neutral

    # following labels are applied
    # negative : 0 
    # positive : 1
    # neutral : 2
    l = {"negative": 0, "positive": 1, "neutral": 2, "objective": 2}
    ext["label"] = [l[label] for label in ext["label"] if label in l]
    f.close()
    
    
    return ext

# helper function that helps identify when we have a new column in our csv
def comma_outta_quote(line):
    in_quote = False
    comma = -1

    count = 0
    while comma == -1 and count < len(line):
        if line[count:count + 1] == "," and not in_quote:
            comma = count
        elif line[count:count + 1] == "\"":
            in_quote = not in_quote

        count += 1
    return comma


if __name__ == "__main__":
    #take_features("data/train.csv")
    #extract("data/train.csv")
    pass
