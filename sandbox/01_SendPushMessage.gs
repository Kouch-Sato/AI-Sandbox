// 参考にした記事
// https://zenn.dev/miya_akari/articles/a8a4c296e7c1c6

const properties = PropertiesService.getScriptProperties();
const kouch_user_id = properties.getProperty('KOUCH_USER_ID');
const access_token = properties.getProperty('LINE_CHANNEL_ACCESS_TOKEN');
const line_push_api = 'https://api.line.me/v2/bot/message/push';

function postMessage() {
  const payload = {
    to: kouch_user_id,
    messages: [
      {
          type: "text",
          text: "いえい！"
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

  UrlFetchApp.fetch(line_push_api, params);
};