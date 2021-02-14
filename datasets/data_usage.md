# Twitter dataset
The Twitter dataset [1] is collected to evaluate social event detection methods. 
After filtering out repeated and irretrievable tweets, the dataset contains 68,841 manually labeled tweets 
related to 503 event classes, spread over a period of four weeks. 
Please find the original dataset at http://mir.dcs.gla.ac.uk/resources/

## Format
The preprocessed Twitter dataset contains the following fields:
```
'event_id': manually labeled event class
'tweet_id': tweet id
'text': content of the tweet
'created_at': timestamp of the tweet
'user_id': the id of the sender
'user_loc', 'place_type', 'place_full_name', and 'place_country_code': the location of the sender
'hashtags': hashtags contained in the tweet
'user_mentions': user mentions contained in the tweet
'image_urls': links to the images contained in the tweet
'entities': a list, named entities in the tweet (extracted using spaCy)
'words': a list, tokens of the tweet (hashtags and user mentions are filtered out)
'filtered_words': a list, lower-cased words of the tweet (punctuations, stop words, hashtags, and user mentions are filtered out)
'sampled_words': a list, sampled words of the tweet (only words that are not in the dictionary are kept to reduce the total number of unique words and maintain a sparse message graph)
```

## Usage
To load preprocessed Twitter dataset, use:

```python
import pandas as pd
import numpy as np

p_part1 = './datasets/Twitter/68841_tweets_multiclasses_filtered_0722_part1.npy'
p_part2 = './datasets/Twitter/68841_tweets_multiclasses_filtered_0722_part2.npy'
df_np_part1 = np.load(p_part1, allow_pickle=True)
df_np_part2 = np.load(p_part2, allow_pickle=True)
df_np = np.concatenate((df_np_part1, df_np_part2), axis = 0)
print("Loaded data.")
df = pd.DataFrame(data=df_np, columns=["event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",\
    "place_type", "place_full_name", "place_country_code", "hashtags", "user_mentions", "image_urls", "entities", 
    "words", "filtered_words", "sampled_words"])
print("Data converted to dataframe.")
print(df.shape)
print(df.head(10))
```

# MAVEN dataset
MAVEN [2] is a general domain event detection dataset constructed from Wikipedia documents. 
We remove sentences (i.e., messages) that are associated with multiple event types. 
The filtered dataset contains 10,242 messages related to 154 event classes.
Please find the original dataset at https://github.com/THU-KEG/MAVEN-dataset

## Format
The preprocessed MAVEN dataset contains the following fields:
```
'message_ids': an unique string for each message, which is a concatenation of 'document_ids' and 'sentence_ids'
'document_ids': an unique string for each document, which corresponds to 'id' in the original MAVEN dataset
'sentence_ids': the index of the sentence in the document, starts from 0, which corresponds to 'sent_id' in the original MAVEN dataset
'sentences': a string, the plain text of the sentence, which corresponds to 'sentence' in the original MAVEN dataset
'event_type_ids': the numerical id for the event type, which corresponds to 'type_id' in the original MAVEN dataset
'words': a list, tokens of the sentence (punctuations and stop words are filtered out)
'unique_words': a list, unique tokens of the sentence (punctuations and stop words are filtered out)
'entities': a list, named entities in the sentence (extracted using spaCy)
```

## Usage
To load preprocessed MAVEN dataset, use:

```python
import pandas as pd
import numpy as np

load_path = './datasets/MAVEN/all_df_words_ents_mids.npy'
df_np = np.load(load_path, allow_pickle=True)
print("Loaded data.")
df = pd.DataFrame(data=df_np, \
    columns=['document_ids', 'sentence_ids', 'sentences', 'event_type_ids', 'words', 'unique_words', 'entities', 'message_ids'])
print("Data converted to dataframe.")
print(df.shape)
print(df.head(10))
```

[1] Andrew J McMinn, Yashar Moshfeghi, and Joemon M Jose. 2013. Building a large-scale corpus for evaluating event detection on twitter. In Proceedings of the CIKM. ACM, 409–418.

[2] Xiaozhi Wang, Ziqi Wang, Xu Han, Wangyi Jiang, Rong Han, Zhiyuan Liu, Juanzi Li, Peng Li, Yankai Lin, and Jie Zhou. 2020. MAVEN: A Massive General Domain
Event Detection Dataset. In Proceedings of EMNLP.
