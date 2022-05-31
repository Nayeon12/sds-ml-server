from flask import Flask, request, jsonify
from werkzeug.serving import WSGIRequestHandler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp
from konlpy.tag import Okt
import scipy as sc

## 문장앞머리에 증상별 숫자태그로 구분

## 단어만 있는경우 제외
# 화, 속앓이, 폭식증, 조각잠, 슬럼프
# 밥밧을 모르겠다 -> 밥맛 오타수정

contents = ['1-머리가 짖눌러지는 느낌', '1-머리가 맑지 않다',
            '1-매일 슬프고 재미없다는 생각', '1-하루종일 울적하다', 
            '1-이유없이 눈물이 많이 난다', '1-며칠이고 하염없이 운다', 
            '1-몇시간씩 운다', '1-눈물이 아무때나 난다', '1-매일 운다',
            '1-외롭다','1-쓸쓸하다', '1-외로움을 잘 탄다',
            '1-친구들과 어울리고 집에오면 외롭다',
            '1-원망스럽다', '1-참을 수 없다', 
            '1-억압하고 살아왔다', '1-늘 내가 참고 지내는 느낌' '1-답답하다',
            '1-괴롭다', '1-화가 치밀어 오르다',
           
            '2-흥이 안난다','2-재미가 없다','2-쳇바퀴 도는 것 같은 일상','2-사는게 재미가 없다','2-사는게 힘겹다',
            '2-기운이 빠지다','2-기운이 없다', '2-늘 기운이 없어보인다',
            '2-기운이 쫙 빠지고','2-막연하게 힘들다','2-간신히 버티고 있다','2-기분이 항상 다운되어있다',
            '2-기분이 가라앉는다','2-축처진다','2-늘어진다',
            '2-귀찮아서 아무것도 못하겠다','2-만사가 다 귀찮다','2-가만히 있고만 싶다',
            '2-아무것도 할 수 없고','2-아무 것도 하기 싫다','2-몸에 힘이 없다',
            '2-혼자 생각에 빠져있을 때가 많다','2-거의 말없이 지냈다','2-혼잣말을 할 때가 있다',
            '2-사람을 피하게 된다','2-사람들 많은 곳 가기 싫다','2-인간관계에 실망스러운 느낌',
            '2-내가 다른 사람들과 매우 다른 것 같다','2-다른 사람의 의도를 이해하기 어렵다','2-공감도 잘 안된다',
            '2-매사에 의욕이 없다','2-매일 의욕이 없다','2-이유없는 무기력감' ,'2-마음이 무겁고 무기력하다','2-집에만 있게된다',
            '2-집에만 있고싶다','2-외출을 거의 하지 않는다','2-외출하기 싫다','2-거의 하루종일 인터넷만 한다',
            '2-하루종일 TV보며 누워있다','2-혼자 누워있는 것 외에 아무것도 하기 싫다',
            
            '3-식욕없다','3-위가 아프다','3-하루에 한 끼 먹는다','3-식욕이 감소하여 체중도 감소','3-체중이 빠지고 머리카락도 많이 빠졌다',
            '3-체중이 빠지고 생리는 6개월 째 안하는 중','3-이유없이 살이 빠지고','3-식욕이 너무 없어서 식욕 촉진제를 먹었다','3-한끼에 반공기정도',
            '3-죽 먹고 있다','3-밥을 먹어야 하나 하는 생각이 든다','3-차려주면 수동적으로 먹는다','3-밥맛을 모르겠다','3-혼자 있으면 거의 안 먹는다 먹고 씹던 음식 버리고 하다가 토하기 시작함',
            '3-보통 하루에 한 번씩 토한다','3-괜찮다가도 갑자기 많이 먹고 토한다',
            '3-예전보다 자꾸 음식이 더 먹고 싶어진다','3-허겁지겁','3-기분이 울적하면 식욕과 상관 없이 폭식','3-식욕증가'
            
            '4-하루종일 잔다','4-밤에 8-10시간 낮에 4시간 잠을 잔다','4-잠은 평소보다 많이 늘었는데 잠을 자고 나도 기분이 좋지 않다','4-하루종일 자고 싶다',
            '4-잠만 자고 싶어서 낮에 약을 먹고 하루에 15시간 정도 잔다','4-밤에 4-5시간 정도 잔다','4-새벽에 일찍 깬다','4-낮에 피곤하다',
            '4-하루 2-3시간 정도밖에 자지 못하고',
            '4-약 먹고 한두시간 잤는데 이젠 아예 잠에 못 들때가 많다','4-수면제를 10알 가량 먹어도 3시간 이상 수면을 지속하기 어려워',
            '4-졸려서 눕는데 말똥..아침에 몸이 무겁고','4-수면제 먹어도 2-3시간 자고','4-새벽에 깨서 더이상 자지 못한다',
            '4-밤에 눈 뜨고 있는 시간이 많다','4-선잠잔 느낌 개운치 않다','4-5~6시간 자지만 자꾸 설친다','4-하루에 6시간 이상 자는데 자주 깬다..일어나도 몸이 무거움',
            '4-약을 먹지 않으면 잠이 안온다','4-30분 수면 후 3시간 깨어있는 상황','4-잠을 잘 자지 못하고 한번 잠을 자면 20시간 가까이 잤다',
            
            '5-늘 쫓기는 느낌','5-불편감','5-잘 모르겠다','5-나 자신을 주체할 수 없다',
            '5-공격적이 된다','5-짜증이 늘었다','5-갑자기 짜증이 난다','5-괜히 짜증이 난다','5-매사가 다 짜증난다',
            '5-신경질이 늘었다','5-스트레스 받으면서 기복 심해진다',
            
            '6-늘 피곤하고 무기력하다','6-아무것도 못할 정도로 피곤한다','6-소진됐다','6-건드려도 반응이 없을 정도로 힘이없다',
            '6-기진맥진하다','6-쉬고싶다','6-몸을 마음대로 움직이기 힘든 느낌',
            
            '7-너무 힘들고 지쳐서 그만두고싶다.','7-이렇게 살면 뭐하나','7-전부 손을 놓게 된다','7-사는게 허무하다',
            '7-모든것을 포기한 느낌','7-기분이 가라앉으면 다 필요없다는 생각이 든다','7-나른하다','7-온 몸에 힘이없다',
            '7-말하는 것도 힘들다','7-의미가 없다','7-지쳤다','7-전부 내려놓고 싶다',
            '7-후회스럽다','7-잘못된 것 같은데 고쳐지지가 않는다',
            '7-걱정이 많이되고, 죄책감이 든다','7-끈기있게 일을 못했다는 죄책감','7-잘 버티지 못한다는 죄책감',
            '7-직장에서 일을 주어진 시간 내에 하지 못하는 것 같아 힘들다','7-내가 뭘 잘못했나 자꾸 생각이 든다',
            '7-자기 자신이 다 잘못한 것 같고, 사람들 만나면 욕하고 안좋은 소리를 할 것 같고','7-그냥 스스로가 점점 죄스럽다',
            
            '8-일에 대한 집중이 어렵다','8-머리가 멍하고 집중도 잘 안된다','8-주의력이 떨어지는 것 같다','8-이해력이 떨어진 것 같다','8-판단력이 서지 않는다','8-정리가 되지 않는다',
            '8-매우 혼란스럽다','8-머릿속이 정리가 안된다','8-제정신이 아니다','8-마음이 정리가 안된다','8-횡설수설','8-생각을 말로 옮기기가 어렵다','8-안정이 안된다',
            '8-모든일이 어렵고 무섭게 느껴졌다','8-어쩔줄을 모르겠다',
            '8-건망증','8-친구 이름도 생각이 안 난 적도 있다','8-기억력 저하','8-기억이 가물가물','8-말이 어눌해 진다',
            '8-한동안 멍하게 무엇을 해야할지 몰라 막막한 기분이 든다','8-몸이 붕 떠있는것 같다','8-기분이 붕 뜰때가 있다',
            
            '9-자신감이 떨어졌다','9-자신이 없고, 못할 것 같은 생각만 든다','9-남들은 다 잘하는데 나만 못하는 것 같다','9-열등생이 된 것 같아 자괴감이 든다',
            '9-패배자 같다','9-자신이 하찮아 보인다','9-주눅들고 자신감이 없다','9-칭찬받은 적이 없다','9-나름대로 열심히 살았는데 뭔가 이루어진 것이 없는것 같다',
            '9-나를 과대평가하는 것 같다','9-주변에서 인정해 줬는데, 실제적으로 결과물이 없어 크게 실망했다','9-다른사람에게 좋은 사람으로 인식되고 싶은데 그러지 못하니까 속상하다',
            '9-어쩔수 없이 내가 하고싶지 않은 일을 하고 있다는 느낌','9-내가 살고싶은대로 살고있지 못하는 것에대한 속상함',
            '9-아무도 자신을 사랑하지 않는 것 같다','9-이전에 내 모습이 없어진 것 같다','9-똑똑하고 이쁜 이미지가 사라진다',
            
            '0-미래에 대해 절망적','0-갈수록 지치고 절망이다. 내가 바보인 것 같다','0-모든 것을 다 잃어버린 것 같다.',
            '0-좋아질거라는 기대감이 안 들고, 살아야 겠다는 생각도 안 든다','0-꿈꿔왔던 길이 막힌다는 기분','0-이미 늦었다',
            '0-이제와서 뭘 새로 할수 있나 하는 생각','0-받아들이기도 힘들고','0-지원을 했었는데 그게 떨어지면서 절망','0-거절을 당하자 절망적인 기분','0-이제는 일도 못 하겠다']

app = Flask(__name__)

@app.route('/', methods = ['POST'])
def Symptom_Detect(contents, input_sen):
  input_sen = request.get_json()
  vectorizer = TfidfVectorizer(min_df=1, decode_error='ignore')
  t = Okt()
  contents_tokens = [t.morphs(row[2:]) for row in contents]
  contents_for_vectorize = []

  for content in contents_tokens:
      sentence = ''
      for word in content:
          sentence = sentence + ' ' + word
      
      contents_for_vectorize.append(sentence)
      
  X = vectorizer.fit_transform(contents_for_vectorize)
  num_samples, num_features = X.shape
  
  vectorizer.get_feature_names()

  new_post_tokens = [t.morphs(row) for row in [input_sen]]
  new_post_for_vectorize = []

  for content in new_post_tokens:
      sentence = ''
      for word in content:
          sentence = sentence + ' ' + word
          
      new_post_for_vectorize.append(sentence)

  new_post_vec = vectorizer.transform(new_post_for_vectorize)
  best_doc = None
  best_dist = 65535
  best_i = None

  for i in range(0, num_samples):
    post_vec = X.getrow(i)

    delta = post_vec - new_post_vec
    d = sp.linalg.norm(delta.toarray())
    
    if d<best_dist:
        best_dist = d
        best_i = i
          
  print("Best post is %i, dist = %.2f" % (best_i, best_dist))
  print('-->', input_sen)

  if(contents[best_i][0] == '1'):
    print("----> 지속되는 우울한 기분 /",contents[best_i][2:])
  elif(contents[best_i][0] == '2'):
    print("----> 흥미나 즐거움의 감소 /",contents[best_i][2:])
  elif(contents[best_i][0] == '3'):
    print("----> 식욕-체중 /",contents[best_i][2:])
  elif(contents[best_i][0] == '4'):
    print("----> 수면 / ",contents[best_i][2:])
  elif(contents[best_i][0] == '5'):
    print("----> 초조나 지체 /",contents[best_i][2:])
  elif(contents[best_i][0] == '6'):
    print("----> 피로나 활력상실 /",contents[best_i][2:])
  elif(contents[best_i][0] == '7'):
    print("----> 무가치하거나 죄책감 /",contents[best_i][2:])
  elif(contents[best_i][0] == '8'):
    print("----> 사고력,집중력감소 또는 우유부단함 /",contents[best_i][2:])
  elif(contents[best_i][0] == '9'):
    print("----> 자존감 저하 /",contents[best_i][2:])
  elif(contents[best_i][0] == '0'):
    print("----> 절망감 /",contents[best_i][2:])


@app.route("/")
def index():
    return "<h1>Welcome to dearmydiary ml server !!</h1>"

if __name__ == "__main__":
    # https://stackoverflow.com/questions/63765727/unhandled-exception-connection-closed-while-receiving-data
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(threaded=True, host='localhost', port=5000)
    '''from waitress import serve
    serve(app, host="0.0.0.0", port=8080)'''