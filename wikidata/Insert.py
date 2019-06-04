import pymysql
from bz2 import BZ2File
import json
import time

start = time.time()

conn = pymysql.connect(
    host='localhost',  # mysql服务器地址
    port=3306,  # 端口号
    user='root',  # 用户名
    passwd='12345678',  # 密码
    db='wikidataplus',  # 数据库名称
    charset='utf8mb4'  # 连接编码
)
cur = conn.cursor()  # 创建并返回游标

sql_1 = "CREATE TABLE IF NOT EXISTS entity(qid VARCHAR(100),qtype VARCHAR(100),qlabel text,qaliase text,qdescription text);"
cur.execute(sql_1)
sql_2 = "CREATE TABLE IF NOT EXISTS property(pid VARCHAR(100),ptype VARCHAR(100),plabel text,paliase text,pdescription text);"
cur.execute(sql_2)
sql_3 = "CREATE TABLE IF NOT EXISTS claim(qid VARCHAR(100),pid VARCHAR(100),datavalue_value text,datavalue_type VARCHAR(100));"
cur.execute(sql_3)


def insert_entity(i, json_object):
    qtype = json_object["type"]
    if qtype == "item":
        qid = json_object['id']

        if 'en' in json_object['labels']:
            qlabel = json_object['labels']['en']['value']
        else:
            qlabel = ''

        if 'en' in json_object['aliases']:
            qaliase = json_object['aliases']['en'][0]['value']
        else:
            qaliase = ''

        if 'en' in json_object['descriptions']:
            qdescription = json_object['descriptions']['en']['value']
        else:
            qdescription = ''

        sql = "insert into entity VALUES(%s,%s,%s,%s,%s)"
        params = (qid, qtype, qlabel, qaliase, qdescription)
        cur.execute(sql, params)
        conn.commit()
    # if i%10000==0:
    print("the %s th line insert entity table ok" % i)


def insert_property(i, json_object):
    global pid, ptype, plabel, paliase, pdescription

    ptype = json_object["type"]
    if ptype == "property":
        pid = json_object["id"]

        if 'en' in json_object['labels']:
            plabel = json_object['labels']['en']['value']
        else:
            plabel = ''

        if 'en' in json_object['aliases']:
            paliase = json_object['aliases']['en'][0]['value']
        else:
            paliase = ''

        if 'en' in json_object['descriptions']:
            pdescription = json_object['descriptions']['en']['value']
        else:
            pdescription = ''

        sql = "insert into property VALUES(%s,%s,%s,%s,%s)"
        params = (pid, ptype, plabel, paliase, pdescription)
        cur.execute(sql, params)
        conn.commit()
    # if i%10000==0:
    print("the %s th line insert property table ok" % i)


def insert_claim(i, json_object):
    global qid, pid, datavalue_value, datavalue_type
    qtype = json_object['type']

    qid = json_object['id']
    for key in json_object['claims']:
        page_object = json_object['claims'][key]
        for key_item in page_object:
            if 'property' in key_item['mainsnak']:
                pid = key_item['mainsnak']['property']
            else:
                pid = ''
            if 'datavalue' in key_item['mainsnak']:
                datavalue = key_item['mainsnak']['datavalue']
            else:
                datavalue = ''
            if datavalue is not '':
                datavalue_value = key_item['mainsnak']['datavalue']['value'] if 'value' in key_item['mainsnak']['datavalue'] and type(
                    key_item['mainsnak']['datavalue']['value']) is str else ''
                datavalue_type = key_item['mainsnak']['datavalue'][
                    'type'] if 'type' in key_item['mainsnak']['datavalue'] else ''

    sql = "insert into claim VALUES(%s,%s,%s,%s)"
    params = (qid, pid, datavalue_value, datavalue_type)
    cur.execute(sql, params)
    conn.commit()
    # if i%10000==0:
    print("the %s th line insert claim table ok" % i)


bz2_file_path = r'./latest-all.json.bz2'
bz2_file = BZ2File(bz2_file_path)


def main():
    i = 1
    count = 1
    for line in bz2_file:
        line_str = line.decode()
        if count < 2:
            print("正在跳过第%s行" % count)
            count += 1
            continue
        if len(line_str) > 2:
            json_object = json.loads(line_str[:-2])
            insert_entity(i, json_object)
            # insert_property(i, json_object)
            insert_claim(i, json_object)
        i += 1
        count += 1
        if count % 100000 == 0:
            print('you have run %s th!' % count)
        # 插入到第count-1行停止，所以下次插入要从第count行开始，即要先跳过count-1行
        if count == 10000001:
            break


if __name__ == '__main__':
    main()
    end = time.time()
    runtime = end-start
    print("running %s" % runtime)
