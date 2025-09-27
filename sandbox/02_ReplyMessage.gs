// 参考にした記事
// https://zenn.dev/miya_akari/articles/cda5e8535833a7


const properties = PropertiesService.getScriptProperties();
const access_token = properties.getProperty('LINE_CHANNEL_ACCESS_TOKEN');
const line_reply_api = 'https://api.line.me/v2/bot/message/reply';

function doPost(e) {
  const event = JSON.parse(e.postData.contents).events[0];
  const replyToken = event.replyToken;
  const userText = event.message.text;

  const payload =  {
    replyToken: replyToken,
    messages: [
      {
          type: "text",
          text: userText,
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