from newspaper import Article
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 텍스트를 문장 단위로 분할
def split_text_into_sentences(text):
    return sent_tokenize(text)

# 문장 간의 유사도 계산
def calculate_sentence_similarity(sentences):
    cv = CountVectorizer()
    vectors = cv.fit_transform(sentences)
    similarity_matrix = cosine_similarity(vectors, vectors)
    return similarity_matrix

# 주요 문장 추출
def extract_summary(text, num_sentences=3):
    sentences = split_text_into_sentences(text)
    if len(sentences) < num_sentences:
        return '문장 수가 요약할 수 있는 문장의 수보다 적습니다.'
    similarity_matrix = calculate_sentence_similarity(sentences)
    sentence_scores = similarity_matrix.sum(axis=1)
    ranked_sentences = [sentence for _, sentence in sorted(zip(sentence_scores, sentences), reverse=True)[:num_sentences]]
    summary = ' '.join(ranked_sentences)
    return summary

# 직접 입력한 문장
input_text = """
한 때 가난한 놀부가 있었습니다. 그는 아이와 아내와 함께 살고 있었는데, 어려움 속에서도 늘 밝은 마음을 간직하고 있었습니다. 그러던 어느 날, 놀부는 밭을 갈기 위해 산에 갔습니다. 그런데 그가 간 동안, 숲 속에 괴물이 나타나서 그의 집을 덮쳤습니다. 괴물은 집 안에 들어가 모든 재산을 빼앗아 갔습니다.

놀부가 집으로 돌아왔을 때, 모든 것이 빼앗겨져 있었습니다. 가난해진 놀부는 마음이 상했지만, 아내와 아이의 안전한 모습을 보며 기운을 내렸습니다. 이후 놀부는 힘든 생활을 견디며 밭일과 목공일로 일을 하며 가족을 먹여 살리기 위해 노력했습니다.

어느 날, 놀부가 산에서 일을 하던 중 갑자기 돌아보니 아름다운 꽃이 한 송이 자라고 있었습니다. 놀부는 이 꽃을 발견하고 꽃의 주변을 흘끗 살펴보았습니다. 그러더니 지면에는 금화가 가득했습니다. 이 꽃이 놀부의 운명을 바꾸는 계기가 되었습니다.

놀부는 이를 토대로 경찰서에 신고하고, 꽃을 발견한 장소를 보여주었습니다. 경찰들은 놀부의 이야기를 의심했지만, 실제로 꽃이 자라고 있던 곳에는 금화가 가득했습니다. 경찰들은 놀부에게서 이 꽃을 발견한 사람의 보상금을 주겠다고 약속했습니다.

그 후, 놀부는 이 꽃으로 많은 돈을 벌게 되었습니다. 놀부는 이 돈을 이용해 집을 다시 짓고, 가족들의 생활을 풍족하게 해주었습니다. 이렇게 놀부는 운명의 꽃을 통해 가난한 삶에서 벗어나 행복한 삶을 살게 되었습니다.
"""

# 추출된 요약문 출력 (10문장 요약)
summary = extract_summary(input_text, num_sentences=3)
print(summary)
