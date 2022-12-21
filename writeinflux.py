# -*- coding: utf-8 -*-
import json
import os

from influxdb import InfluxDBClient

from influxdb import DataFrameClient
class WriteInflux:

    def __init__(self, data):
        """Constructor"""
        self.data = data
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

        with open(ROOT_DIR+'/config.json') as f:
            data_config = json.load(f)
        config_influx = data_config.get('influx')
        self.config_influx = config_influx
        self.database = config_influx['database']
        if self.connect_influx():
            client = self.connect_influx()
            self.check_db(client)
            self.export_to_influxdb(client)
            client.close()





    def connect_influx(self):
        host = self.config_influx['host']
        port = self.config_influx['port']
        client = InfluxDBClient('localhost', 8086)

        try:
            client.ping()
        except Exception as e:
            print(e)
            return False
        return client

    def check_db(self, client):
        database_name = self.database
        if not ({'name': database_name} in client.get_list_database()):
            client.create_database(database_name)
        client.switch_database(database_name)

        """
        Запись в базу данных временных рядов
        :param polling_time_value:
        :return:
        """

    def export_to_influxdb(self, client):

        json_body = self.data
        print(json_body)
        client.write_points(json_body,time_precision="ms")
    def read_data(self,client):
        print(client.query('SELECT * FROM Electro'))