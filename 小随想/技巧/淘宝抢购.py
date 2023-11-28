# -*- coding: utf-8 -*-
import datetime
import time

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains


def logintaobao(browser):
    browser.get("https://www.taobao.com")
    print("请等待3s网页加载完全\n")
    time.sleep(3)
    if browser.find_element_by_link_text("亲，请登录"):
        browser.find_element_by_link_text("亲，请登录").click()
        print("请30s内完成账户登录\n")
        time.sleep(30)


def picking(browser, method):
    browser.get("https://cart.taobao.com/cart.htm")
    print(f"请等待3s网页加载完全")
    time.sleep(3)
    if method == 0:
        while True:
            try:
                if browser.find_element_by_id("J_SelectAll1"):
                    browser.find_element_by_id("J_SelectAll1").click()
                    break
            except Exception as e:
                print("输出一条异常信息:\n", e)
                print("再次尝试全选\n")
    else:
        print("请15s内动勾选需要购买的商品\n")
        time.sleep(15)


def buy(times):
    even = 1
    while True:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        if now >= times:
            while True:
                try:
                    if browser.find_element_by_id("J_Go"):
                        tmp = browser.find_element_by_id("J_Go")
                        ActionChains(browser).move_to_element(tmp).perform()
                        browser.find_element_by_id("J_Go").click()
                        print(f"结算成功，准备提交订单\n")
                        break
                except Exception as e:
                    print("输出一条异常信息:\n", e)
                    break

            while True:
                try:
                    if browser.find_element_by_link_text("提交订单"):
                        browser.find_element_by_link_text("提交订单").click()
                        print("抢购成功，请尽快付款\n")
                        break
                except Exception as e:
                    print("输出一条异常信息:\n", e)
                    break

            time.sleep(5)
            print("再次尝试\n若已经成功请暂停程序\n若多次失败请暂停程序\n")

        else:
            if even:
                print("即将开始，耐心等待，当前时间是:{now}\n")
                even = 0


if __name__ == "__main__":
    browser = webdriver.Chrome()
    logintaobao(browser)
    method = int(input("请输入模式0或者1\n(0:自动全选商品；1:手动选择商品)\n:"))
    picking(browser, method)
    times = input("请输入抢购时间\n(例如2021-06-01 12:00:00.000000)\n:")
    buy(times)
