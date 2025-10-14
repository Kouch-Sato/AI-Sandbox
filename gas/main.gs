const properties = PropertiesService.getScriptProperties();
const accessToken = properties.getProperty('LINE_CHANNEL_ACCESS_TOKEN');
const openaiKey = properties.getProperty('OPENAI_API_KEY');
const sheetId = properties.getProperty('SHEET_ID');
const kouchUserId = properties.getProperty('KOUCH_LINE_USER_ID');
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

  const chatHistory = getChatHistory(userId);
  const extractedTrigger = extractTriggerWithGPT(userText, chatHistory);
  const replyText = generateTextWithGPT(userText, chatHistory, extractedTrigger);

  if (extractedTrigger) {
    logTriggerToSheet(userId, extractedTrigger);
  }

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

function generateTextWithGPT(userText, chatHistory, extractedTrigger) {
  const payload = {
    model: 'gpt-5-chat-latest',
    messages: [
      { role: 'system', content: replyChatPrompt },
      { role: 'system', content: "extractTrigger: " + JSON.stringify(extractedTrigger) },
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

function extractTriggerWithGPT(userText, chatHistory) {
  const payload = {
    model: 'gpt-5-chat-latest',
    messages: [
      { role: 'system', content: extractTriggerPrompt },
      ...chatHistory,
      { role: 'user',   content: userText }
    ],
    max_tokens: 200,
    temperature: 0.1,
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
  const text = (json && json.choices && json.choices[0] && json.choices[0].message && json.choices[0].message.content) || '';
  return pickJsonOrNull(text);
};

function pickJsonOrNull(text) {
  if (!text) return null;
  const s = String(text).trim();

  // ```json ... ``` や ``` ... ``` を剥がす
  const fenced = s.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
  const raw = fenced ? fenced[1].trim() : s;

  // 先頭の { ... } だけ抜き出す or null
  const objMatch = raw.match(/{[\s\S]*}/);
  const token = objMatch ? objMatch[0] : (raw === 'null' ? 'null' : null);
  if (!token) return null;

  try {
    return JSON.parse(token);
  } catch (_) {
    return null;
  }
};

function logTriggerToSheet(userId, trigger) {
  const sheet = getSheet('trigger');
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(['timestamp', 'user_id', 'when_iso', 'action', 'reminder_text']);
  }

  if (trigger) {
    sheet.appendRow([new Date(), userId, trigger.when_iso, trigger.action, trigger.reminder_text]);
  }
};
