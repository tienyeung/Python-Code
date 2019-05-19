import pkuseg
# from collections import Counter
# import pprint
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


content = []
with open('/home/yeung/PycharmProjects/untitled/wordcloud/shijiuda.txt', encoding='utf-8') as f:
        content = f.read()

lexicon=['习近平', '中国']
seg = pkuseg.pkuseg(user_dict=lexicon)
text = seg.cut(content)

stopwords = []
with open('/home/yeung/PycharmProjects/untitled/wordcloud/stopword.txt', encoding='utf-8') as f:
    stopwords = f.read()

new_text = ''
for w in text:
    if w not in stopwords:
        new_text += w
        new_text += " "



# counter = Counter(new_text)
# speech = list(counter.elements())
# print(speech)
# pprint.pprint(counter.most_common(50))

img = Image.open('/home/yeung/PycharmProjects/untitled/wordcloud/party.png')
img_array = np.array(img)
wordcloud = WordCloud(background_color='white', font_path='usr/share/fonts/YaHeiConsolas.ttf', mask=img_array).generate(new_text)
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.figure()


wordcloud.to_file('hah.png')

