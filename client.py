import socket

from sympy import false

import voice2text
import torch
from pathlib import Path
from qwen_asr import Qwen3ASRModel
import TTS
import myTools
import actions


HOST = "127.0.0.1" # 服务端 IP（本机测试用 127.0.0.1）
PORT = 8888


def main():
    text=""
    input_mode = 2 # 切换输入模式
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    home_status = actions.HomeStatus() # 初始化状态机
    # ===== 加载 Qwen3-ASR 模型 =====
    print("正在加载 Qwen3-ASR 模型...")
    model_dir = Path("voice2text/Qwen3-ASR-1___7B")
    asr_model = Qwen3ASRModel.from_pretrained(
        str(model_dir),
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=512,
    )
    print("模型加载完成！")
    try:
        """开始运行的时候就先建立连接"""
        client_socket.connect((HOST, PORT))
        print(f"[客户端] 已连接到 {HOST}:{PORT}")
        is_connected = True
        # 登录
        login = false
        while not login:
            if not is_connected:
                # 如果之前有断开过连接就在while循环开始的时候重新建立
                client_socket.connect((HOST, PORT))
                print(f"[客户端] 已连接到 {HOST}:{PORT}")
                is_connected = True
            print("请问是要登录还是注册：（1.登录，2.注册）")
            num = input()
            print("请输入用户名：")
            username = input()
            print("请输入密码：")
            password = input()
            print("请输入角色名：")
            character_name = input()
            if num == "1":
                client_socket.sendall("login$".encode("utf-8"))
                client_socket.sendall((username + "$").encode("utf-8"))
                client_socket.sendall((password + "$").encode("utf-8"))
                client_socket.sendall((character_name + "$").encode("utf-8"))
                recv_buf = b""
                result, recv_buf = myTools.get_a_message(client_socket, recv_buf)
                if result == "login_success":
                    login = True
                    print("登录成功")
                elif result == "password_uncorrected":
                    print("密码错误")
                elif result == "user_not_exists":
                    print("用户名不存在")
                elif result == "character_not_exist":
                    print("登录时所请求的角色不存在")
                # 这个是由服务器抛出的自定义异常返回的字符串，所以不管是注册还是登录都需要处理
                elif result == "ClientDisconnected":
                    print("连接意外断开")
            elif num == "2":
                client_socket.sendall("setup$".encode("utf-8"))
                client_socket.sendall((username + "$").encode("utf-8"))
                client_socket.sendall((password + "$").encode("utf-8"))
                recv_buf = b""
                result, recv_buf = myTools.get_a_message(client_socket, recv_buf)
                if result == "setup_success":
                    print("账号创建成功")
                elif result == "username_exist":
                    print("用户已存在，请登录")
                elif result == "new_user_not_exist":
                    print("new_user文件夹不存在")
                elif result == "setup_unsuccess":
                    print("创建账号失败")
                # 这个是由服务器抛出的自定义异常返回的字符串，所以不管是注册还是登录都需要处理
                elif result == "ClientDisconnected":
                    print("连接意外断开")
        recv_buf = b""
        memory_management_is_success, recv_buf = myTools.get_a_message(client_socket, recv_buf)
        if memory_management_is_success == "memory_management_unsuccess":
            print("记忆管理失败，服务器端的是否请求和聊天记录的条数对不上")
            client_socket.close() # 尽管服务器已经关闭了连接也还是再客户端这边关闭一次，规范代码书写
            return

        """**********************************************************************************************************"""
        """从这里开始是登录成功之后可以执行的所有逻辑，都写入一个大的while循环"""
        """整个部分的主要逻辑如下"""
        """1.先判断用户要做什么操作（req）"""
        """2.根据该操作向用户获取所需的信息，有一些发给服务器，有一些保存到客户端本地"""
        """3.接收服务器反馈并做相应处理"""
        """4.如果是连接断开就直接关闭客户端"""
        while True:
            print("请输入需要向服务器发送的请求：（chat：聊天、character：创建新角色、change_character：切换聊天角色、quit：退出）")
            req=input()



            if req == "1":
                if input_mode == 1:
                    text = input("请输入: ")   # 读取键盘输入
                    if text == "":
                        print("text为空")
                        continue
                elif input_mode == 2:
                    print("正在努力倾听您的话：")
                    text = voice2text.v2t(asr_model)
                print(f"玉舟哥哥说：{text}")
                # 发送文本，必须以 $ 结尾
                client_socket.sendall("chat$".encode("utf-8"))
                client_socket.sendall((text + "$").encode("utf-8"))
                client_socket.sendall((home_status.to_prompt() + "$").encode("utf-8"))
                # recv_buf必须定义在外层，不然会丢失信息
                recv_buf = b""
                reply,recv_buf = myTools.get_a_message(client_socket,recv_buf)
                control_code_str,recv_buf = myTools.get_a_message(client_socket,recv_buf)
                sentiment_label,recv_buf = myTools.get_a_message(client_socket,recv_buf) # 接收情感标签
                sentiment_confidence,recv_buf = myTools.get_a_message(client_socket,recv_buf) # 接收情感标签可信度（这是一个浮点数，需要做类型转换）
                print(f"由比滨结衣：{reply}")
                print(f"控制代码：{control_code_str}")
                print(f"情感标签：{sentiment_label}")
                print(f"情感可信度{sentiment_confidence}")
                if control_code_str!="":
                    for ch in control_code_str:
                        code = int(ch)
                        actions.do_action(home_status, code)
                    print(home_status.to_prompt())
                # ---------- 调用语音生成 ----------
                segments = myTools.split_by_chinese(reply, limit=12)
                TTS.segments2voice(segments,"happiness",sentiment_confidence)


                """目前，创建角色和切换角色都仅仅完成了服务器端的切换，客户端还需要开发对应的切换,例如：语音模型"""
            elif req == "2":
                print("请输入角色姓名：")
                new_character_name = input()
                print("请输入角色设定")
                character_set = input()
                client_socket.sendall("create_a_new_character$".encode("utf-8"))
                client_socket.sendall((new_character_name + "$").encode("utf-8"))
                client_socket.sendall((character_set + "$").encode("utf-8"))
                recv_buf = b""
                create_success, recv_buf = myTools.get_a_message(client_socket, recv_buf)
                if create_success=="create_success":
                    print(f"新角色{new_character_name}创建成功")
                elif create_success=="create_unsuccess":
                    print(f"新角色{new_character_name}创建失败")
                elif create_success=="infor_error":
                    print("角色信息错误")



            elif req == "3":
                print("请输入需要切换的角色名称：")
                change_name = input()
                client_socket.sendall("change_character$".encode("utf-8"))
                client_socket.sendall((change_name + "$").encode("utf-8"))
                recv_buf = b""
                result, recv_buf = myTools.get_a_message(client_socket, recv_buf)
                if result == "change_success":
                    print("角色切换成功")
                elif result == "change_unsuccess":
                    print("角色切换失败")



            elif req == "quit":
                print("退出")
                recv_buf = b""
                quit_success, recv_buf = myTools.get_a_message(client_socket, recv_buf)
                if quit_success=="quit_success":
                    print("退出成功")



    except ConnectionRefusedError:
        print("[客户端] 连接被拒绝，请确认服务端已启动")
    except KeyboardInterrupt:
        print("\n[客户端] 手动中断")
    except myTools.ClientDisconnected:
        print("连接意外断开")
    finally:
        client_socket.close()
        print("[客户端] 已关闭")


if __name__ == "__main__":
    main()