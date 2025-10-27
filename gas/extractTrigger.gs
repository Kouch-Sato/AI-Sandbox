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
