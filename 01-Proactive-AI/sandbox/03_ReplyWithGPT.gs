// 参考にした記事
// https://zenn.dev/miya_akari/articles/cda5e8535833a7


const properties = PropertiesService.getScriptProperties();
const access_token = properties.getProperty('LINE_CHANNEL_ACCESS_TOKEN');
const openai_key = properties.getProperty('OPENAI_API_KEY');
const line_reply_api = 'https://api.line.me/v2/bot/message/reply';
const openai_api = 'https://api.openai.com/v1/chat/completions';

function doPost(e) {
  const event = JSON.parse(e.postData.contents).events[0];
  const replyToken = event.replyToken;
  const userText = event.message.text;

  const replyText = generateTextWithGPT(userText);

  const payload =  {
    replyToken: replyToken,
    messages: [
      {
          type: "text",
          text: replyText,
      }
    ]
  };

  const params = {
    method: 'post',
    contentType: 'application/json; charset=UTF-8',
    headers: {
      Authorization: 'Bearer ' + access_token
    },
    payload: JSON.stringify(payload)
  };

  UrlFetchApp.fetch(line_reply_api, params);
};

function generateTextWithGPT(userText) {
    const payload = {
      model: 'gpt-5-chat-latest',
      messages: [
        { role: 'system', content: 'あなたはフレンドリーな解説員です。ユーザーの入力に対して、その単語の説明やちょっとした豆知識を200文字程度で返してください。 返答は口語的でその単語に関する内容だけ返してください。 （わかりました、などの返事は不要です）' },
        { role: 'user',   content: userText }
      ],
      // 必要に応じて:
      // temperature: 0.7,
      max_tokens: 200,
    };

    const params = {
      method: 'post',
      contentType: 'application/json; charset=UTF-8',
      headers: {
        Authorization: 'Bearer ' + openai_key
      },
      payload: JSON.stringify(payload)
    };

    const res = UrlFetchApp.fetch(openai_api, params);

    const json = JSON.parse(res.getContentText());
    // Chat Completionsの返却形式：choices[0].message.content
    const text = (json && json.choices && json.choices[0] && json.choices[0].message && json.choices[0].message.content) || '（すみません、うまく生成できませんでした）';
    return text.trim();
};
