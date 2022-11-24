import re
from googleapiclient.discovery import build
import json
import time
from datetime import datetime, timedelta
from transformers import ElectraForSequenceClassification, ElectraTokenizerFast

api_key = ''
video_id = ''

api_obj = build('youtube', 'v3', developerKey=api_key)
response = api_obj.commentThreads().list(part="id, replies, snippet", videoId=video_id, maxResults=100).execute()

cur_index = 0
result = []

while response:
    ### datatype "0" 원댓글
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']

        result_format = {
            "index": cur_index,
            "datatype": "0",
            "toWho": "",
            "author": comment['authorDisplayName'],
            "publishedDate": comment['publishedAt'],
            "timeNum": re.sub(r'[^0-9]', '', comment['publishedAt']),
            "text": comment['textDisplay']
        }
        result.append(result_format)
        cur_index = cur_index + 1

        ### datatype "1", "2" 대댓글
        if 'replies' in item.keys():
            reply_num = 0
            parent = item['id']
            request = api_obj.comments().list(part='snippet', parentId=parent, maxResults=100).execute()
            while request:
                for reply in request['items']:
                    reply['snippet']['textDisplay'] = reply['snippet']['textDisplay'].lstrip()

                    ### datatype "2" 언급대댓글
                    if reply['snippet']['textDisplay'][0] == '@':
                        reference = reply['snippet']['textDisplay'].split(' ', 1)
                        reference = reference[0].replace("@", "")

                        close_reference_index = -1
                        close_reference = []
                        reference_toWho = 0

                        ### 언급한 댓글 찾는 과정
                        for re_reply in request['items']:
                            close_reference_index = close_reference_index + 1 # 대댓글 내 target 대댓글 index
                            if re_reply['snippet']['authorDisplayName'] == reference: # 언급자와 target 대댓글 author가 같을 경우
                                close_reference.append(close_reference_index) # close_reference에 target 대댓글 index 추가

                        if len(close_reference) == 0: # 언급자와 일치하는 대댓글 내 author가 없을 시
                            if reference == comment['authorDisplayName']: # 원댓글 author와 언급자가 동일한지
                                reference_toWho = cur_index - reply_num # 동일하면 reference_toWho = 원댓글 index
                            else:
                                # 언급자 존재 X | 언급자 이름 내 띄쓰 존재
                                reference_toWho = ""
                        else:
                            close_reference_list = []
                            for i in range(0, len(close_reference)):
                                if reply_num < close_reference[i]: # 언급 대댓글이 target 대댓글보다 나중에 달렸을 시(최신순으로 읽어옴)
                                    close_reference_list.append(abs(reply_num - close_reference[i])) # close_reference_list에 언급 대댓글과 target 대댓글 거리 차이 추가
                                else:
                                    close_reference_list.append(99999)
                            if len(close_reference_list) == 0: # 언급 대댓글이 target 대댓글보다 먼저 달렸을 시 (아마 언급자가 먼저 달았던 댓글을 삭제한 경우)
                                # 언급 대댓글 후 target 존재
                                reference_toWho = ""
                            else:
                                close_index = min(close_reference_list) # close_reference_list 내에 가장 작은 값 = 언급 대댓글과 가장 가까운 값 추출
                                reference_toWho = cur_index + close_index # 해당 index reference_toWho에 추가

                        result_format = {
                            "index": cur_index,
                            "datatype": "2",
                            "toWho": reference_toWho,
                            "author": reply['snippet']['authorDisplayName'],
                            "publishedDate": reply['snippet']['publishedAt'],
                            "timeNum": re.sub(r'[^0-9]', '', reply['snippet']['publishedAt']),
                            "text": reply['snippet']['textDisplay']
                        }
                        result.append(result_format)
                        cur_index = cur_index + 1
                        reply_num = reply_num + 1

                    ### datatype "1" 대댓글
                    else:
                        result_format = {
                            "index": cur_index,
                            "datatype": "1",
                            "toWho": cur_index - reply_num - 1,
                            "author": reply['snippet']['authorDisplayName'],
                            "publishedDate": reply['snippet']['publishedAt'],
                            "timeNum": re.sub(r'[^0-9]', '', reply['snippet']['publishedAt']),
                            "text": reply['snippet']['textDisplay']
                        }
                        result.append(result_format)
                        cur_index = cur_index + 1
                        reply_num = reply_num + 1

                if 'nextPageToken' in request:
                    request = api_obj.comments().list(part='snippet', parentId=parent, pageToken=request['nextPageToken'],
                                                      maxResults=100).execute()
                else:
                    break

    if 'nextPageToken' in response:
        response = api_obj.commentThreads().list(part='snippet,replies', videoId=video_id,
                                                 pageToken=response['nextPageToken'], maxResults=100).execute()
    else:
        break
print("len1: ", len(result))

### 광고도배(중복), html 태그, 5글자 미만 댓글 삭제
del_set = set()
for i in range(0, len(result)):
    result[i]['text'] = result[i]['text'].replace('<br>', ' ').replace('\r', ' ') # <br>, \r 제거
    result[i]['text'] = re.sub('<a.*a>', '', result[i]['text']) # <a href= ~~> 제거
    curr_text = result[i]['text']
    curr_author = result[i]['author']
    for j in range(i+1, len(result)):
        target_text = result[j]['text']
        target_author = result[j]['author']
        if curr_text == target_text and curr_author == target_author:
            del_set.add(i)
            del_set.add(j)
        # text 5 미만 삭제
        elif len(curr_text) < 5:
            del_set.add(i)


for i in del_set:
    for j in range(0, len(result)-1):
        if result[j]['index'] == i:
            result.pop(j)

print("len2: ", len(result))

### 시간순 정렬
result = sorted(result, key=lambda x: x['timeNum'], reverse=False)

### 6시간 5개 이하 삭제
std = 0
TF = True
while TF:
    j = 0
    dt = datetime.strptime(result[std]['publishedDate'], '%Y-%m-%dT%H:%M:%SZ')
    std_dt = dt + timedelta(hours=6)
    std_timestamp = time.mktime(std_dt.timetuple())
    for i in range(std, len(result)):
        target_dt = datetime.strptime(result[i]['publishedDate'], '%Y-%m-%dT%H:%M:%SZ')
        target_timestamp = time.mktime(target_dt.timetuple())
        if target_timestamp <= std_timestamp:
            j = j + 1
        else:
            if j <= 5:
                while std < len(result):
                    result.pop(std)
                TF = False
            else:
                std = std + j
            break

print("len3: ", len(result))


######################## 감성분석 ##########################

args = {
    'train_data_path': './ratings_train.txt',
    'val_data_path': './ratings_test.txt',
    'save_path': './model',
    'max_epochs': 1,
    'model_path': 'beomi/KcELECTRA-base',
    'batch_size': 32,
    'learning_rate': 5e-5,
    'warmup_ratio': 0.0,
    'max_seq_len': 128
}

model = ElectraForSequenceClassification.from_pretrained(args['save_path'])
tokenizer = ElectraTokenizerFast.from_pretrained(args['model_path'])

### 한 줄씩 감성 분석
for i in result:
    try:
        input_vector = tokenizer.encode(i['text'], return_tensors='pt')
        pred = model(input_ids=input_vector, labels=None).logits.argmax(dim=-1).tolist()
    except:
        continue
    i['score'] = pred[0]

### 파일 저장
with open('파일이름.json', 'w', encoding='utf-8') as make_file:
    json.dump(result, make_file, ensure_ascii=False, indent='\t')
