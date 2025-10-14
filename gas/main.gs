const properties = PropertiesService.getScriptProperties();
const accessToken = properties.getProperty('LINE_CHANNEL_ACCESS_TOKEN');
const openaiKey = properties.getProperty('OPENAI_API_KEY');
const sheetId = properties.getProperty('SHEET_ID');
const lineReplyApi = 'https://api.line.me/v2/bot/message/reply';
const openaiApi = 'https://api.openai.com/v1/chat/completions';

function getSheet(name) {
  const sheet = SpreadsheetApp.openById(sheetId).getSheetByName(name);
  return sheet;
}

function doPost(e) {
  const event = JSON.parse(e.postData.contents).events[0];
  const replyToken = event.replyToken;
  const userText = event.message.text;
  const userId = event.source.userId;

  logChatToSheet(userId, 'user', userText);

  const replyText = generateTextWithGPT(userId, userText);

  logChatToSheet(userId, 'assistant', replyText);

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
      Authorization: 'Bearer ' + accessToken
    },
    payload: JSON.stringify(payload)
  };

  UrlFetchApp.fetch(lineReplyApi, params);
};

function generateTextWithGPT(userId, userText) {
  const chatHistory = getChatHistory(userId);

  const payload = {
    model: 'gpt-5-chat-latest',
    messages: [
      { role: 'system', content: replyChatPrompt },
      ...chatHistory,
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
      Authorization: 'Bearer ' + openaiKey
    },
    payload: JSON.stringify(payload)
  };

  const res = UrlFetchApp.fetch(openaiApi, params);

  const json = JSON.parse(res.getContentText());
  // Chat Completionsの返却形式：choices[0].message.content
  const text = (json && json.choices && json.choices[0] && json.choices[0].message && json.choices[0].message.content) || '（すみません、うまく生成できませんでした）';
  return text.trim();
};
