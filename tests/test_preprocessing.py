import unittest
import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType
import tracemalloc
tracemalloc.start()
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.data.preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = SparkSession.builder.appName("TestPreprocessing").getOrCreate()
        cls.preprocessor = DataPreprocessor(cls.spark)

        # Create a sample DataFrame for testing
        schema = StructType([
            StructField("did", StringType(), False),
            StructField("ip", StringType(), True),
            StructField("fqdn", StringType(), True),
            StructField("puerto", IntegerType(), True),
            StructField("fingerprint", StringType(), True),
            StructField("data", StringType(), True),
            StructField("cpe", StringType(), True),
            StructField("cve", ArrayType(StringType()), True),
            StructField("vectores", ArrayType(StringType()), True)
        ])

        data = [
            ("1", "192.168.1.1", "example.com", 80, "Apache/2.4.41", "Some data here", "cpe:/a:apache:http_server:2.4.41", ["CVE-2021-12345"], ["WEB-VULN-01"]),
            ("2", "10.0.0.1", None, 443, "NGINX 1.18.0", "", None, [], []),
            ("3", "172.16.0.1", "test.local", 22, "OpenSSH_7.9", "SSH-2.0-OpenSSH_7.9", "cpe:/a:openbsd:openssh:7.9", ["CVE-2020-54321"], ["NET-VULN-02"])
        ]

        cls.test_df = cls.spark.createDataFrame(data, schema)

    def test_preprocess(self):
        processed_df = self.preprocessor.preprocess(self.test_df)
        
        # Check if the new 'combined_text' column is created
        self.assertIn('combined_text', processed_df.columns)
        
        # Check if the number of rows remains the same
        self.assertEqual(self.test_df.count(), processed_df.count())
        
        # Check if IP addresses are formatted correctly
        ip_sample = processed_df.select('ip').first()[0]
        self.assertFalse('.' in ip_sample)

        # Check if 'puerto' is converted to string
        puerto_type = processed_df.schema['puerto'].dataType
        self.assertIsInstance(puerto_type, StringType)

        # Check if text fields are cleaned
        fingerprint_sample = processed_df.select('fingerprint').first()[0]
        self.assertEqual(fingerprint_sample, "apache/2.4.41")

        # Check if combined_text contains all necessary information
        combined_text_sample = processed_df.select('combined_text').first()[0]
        self.assertIn("IP:", combined_text_sample)
        self.assertIn("Port:", combined_text_sample)
        self.assertIn("Fingerprint:", combined_text_sample)

    def test_clean_text(self):
        test_cases = [
            ("Apache/2.4.41", "apache/2.4.41"),
            ("NGINX 1.18.0", "nginx 1.18.0"),
            ("", ""),
            (None, "")
        ]
        for input_text, expected_output in test_cases:
            self.assertEqual(self.preprocessor.clean_text(input_text), expected_output)

    def test_combine_fields(self):
        sample_row = self.test_df.first()
        combined_text = self.preprocessor.combine_fields(
            sample_row['ip'], sample_row['fqdn'], str(sample_row['puerto']),
            sample_row['fingerprint'], sample_row['cpe'], sample_row['cve'], sample_row['vectores']
        )
        self.assertIn("IP: 192.168.1.1", combined_text)
        self.assertIn("FQDN: example.com", combined_text)
        self.assertIn("Port: 80", combined_text)
        self.assertIn("Fingerprint: Apache/2.4.41", combined_text)
        self.assertIn("CPE: cpe:/a:apache:http_server:2.4.41", combined_text)
        self.assertIn("CVEs: CVE-2021-12345", combined_text)
        self.assertIn("Vectors: WEB-VULN-01", combined_text)

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

if __name__ == '__main__':
    unittest.main()