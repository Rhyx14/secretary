from .sys_info import get_host_name
import requests
def info_wechat_autodl(token,title,name=None,content=None):    
    # python脚本示例
    if(name is None):
        name=get_host_name()
    if content is None:
        content='no content'
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                 json={
                     "token": token,
                     "title": title,
                     "name": name,
                     "content": content
                 })