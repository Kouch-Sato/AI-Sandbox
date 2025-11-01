const line_push_api = 'https://api.line.me/v2/bot/message/push';

// GASの設定により、この関数は毎分実行されます。
function fireTrigger() {
  const sheet = getSheet('trigger');
  const data = sheet.getDataRange().getValues();

  const now = new Date();
  const oneMinuteLater = new Date(now.getTime() + 1 * 60 * 1000);

  // rowの形式: [timestamp, user_id, when_iso, action, reminder_text]
  const triggersToFire = data.filter(row => {
    if (row[0] === 'timestamp') return false;
    
    const [timestamp, userId, whenIso, action, reminderText] = row;
    const triggerTime = new Date(whenIso);
    return triggerTime >= now && triggerTime < oneMinuteLater;
  });

  triggersToFire.forEach(row => {
    const [timestamp, userId, whenIso, action, reminderText] = row;
    postMessage(userId, reminderText);
  });
}

function postMessage(userId, reminderText) {
  const payload = {
    to: userId,
    messages: [
      {
        type: "text",
        text: reminderText,
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

  UrlFetchApp.fetch(line_push_api, params);
}
