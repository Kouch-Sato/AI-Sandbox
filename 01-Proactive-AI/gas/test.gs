const kouchUserId = properties.getProperty('KOUCH_LINE_USER_ID');; // 自分のLINE USER IDを入れる

function testDoPost() {
 const e = {
    postData: {
      contents: JSON.stringify({text: "hello"}),
      length: 17,
      type: 'application/json'
    },
    parameter: { foo: 'bar' }
  };

  const response = doPost(e);
  Logger.log(response.getContent());
} 

// 最小の e を作る（テキストメッセージ1件）
function buildE_Min({ userId = kouchUserId, text = '10/5 9:00にリマインドして', replyToken = 'dummy-reply' } = {}) {
  const contents = JSON.stringify({
    events: [{
      replyToken,
      source: { userId },              // 使うのは userId だけ（typeは不要）
      message: { type: 'text', text }  // 使うのは text だけ
    }]
  });
  return { postData: { contents } };
}

// 使い方（doPostを直接呼ぶ）
function test_doPost_min() {
  const e = buildE_Min({ userId: 'TEST_USER_001', text: '10/5 9:00にリマインドして' });
  const res = doPost(e);
  Logger.log(res && res.getContent && res.getContent());
}
