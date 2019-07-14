from selenium import webdriver
# from selenium.webdriver.common.keys import Keys

browser = webdriver.Firefox('http://gmail.com')
emailelem = browser.find_element_by_id('Email')
emailelem.send_keys('yeungtien@gmail.com')
passwordelem = browser.find_element_by_id('Passwd')
passwordelem.send_keys('Ty10254416')
passwordelem.submit()
# passwordelem.send_keys(Keys.ENTER)
