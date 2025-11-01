function logChatToSheet(userId, direction, text) {
  const sheet = getSheet('chat_log');
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(['timestamp', 'user_id', 'direction', 'text']);
  }

  sheet.appendRow([new Date(), userId, direction, text]);
}

function getChatHistory(userId) {
  const sheet = getSheet('chat_log');
  const data = sheet.getDataRange().getValues();

  const userChat = data.filter(row => row[1] === userId);

  // userChatの各行は [timestamp, user_id, direction, text] の形式
  // これをChatGPTの会話履歴形式に変換
  // directionが'user'か'assistant'に対応
  // 例:
  // { role: "assistant", content: "アシスタントの性格や目的の指定" },
  // { role: "user", content: "ユーザーの発言" },
  // { role: "assistant", content: "モデルの返答" },
  // { role: "user", content: "次の発言" },

  const chatHistory = userChat.map(row => {
    return { role: row[2], content: row[3] };
  });

  return chatHistory;
}
