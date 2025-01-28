import json
import os
from pathlib import Path
import requests
import subprocess
import tempfile
import time

import psutil

ENV_VAR_FUSEKI_PATH = 'FUSEKI_HOME'
ENV_VAR_JENA_PATH = 'JENA_HOME'
DEFAULT_FUSEKI_PORT = 3030


class LocalFusekiServer:
    __ports_in_use: set = set()

    def __init__(self, graph_path: Path, port: int = None, timeout: int = None):
        self.graph_path = Path(graph_path).resolve()
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {self.graph_path}")
        if port in self.__ports_in_use:
            raise ValueError(f"Port {port} is already used")

        self.fuseki_dataset = self.graph_path.stem
        self.port = port if port is not None else DEFAULT_FUSEKI_PORT
        self.fuseki_endpoint = f'http://localhost:{self.port}/{self.fuseki_dataset}'
        self.fuseki_home_path = Path(os.environ.get(ENV_VAR_FUSEKI_PATH)).resolve()
        if not self.fuseki_home_path.exists():
            raise EnvironmentError(f"Invalid Fuseki path: {self.fuseki_home_path}")

        # Start the Fuseki server
        self.timeout = timeout
        cmd_list = [
            str(self.fuseki_home_path.joinpath('fuseki-server.bat')),
            f'--file={self.graph_path}',
            f'--port={self.port}',
            '--localhost',
            f'--timeout={self.timeout}' if self.timeout is not None else '',
            f'/{self.fuseki_dataset}'
        ]
        self.process = subprocess.Popen(
            cmd_list,
            cwd=self.fuseki_home_path,
            creationflags=subprocess.DETACHED_PROCESS
        )
        self.__ports_in_use.add(self.port)

    def query(self, query: str, timeout: int = None, retries: int = 3) -> dict:
        if self.process.poll() is not None:
            raise RuntimeError(f"Server crashed on endpoint: {self.fuseki_endpoint}")
        while True:
            try:
                response = requests.post(
                    f'{self.fuseki_endpoint}/sparql',
                    data={'query': query},
                    headers={'Accept': 'application/json'},
                    timeout=timeout if timeout is not None else self.timeout
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if retries >= 1:
                    time.sleep(1)
                    retries -= 1
                else:
                    raise e

    def stop(self):
        for child in psutil.Process(self.process.pid).children(recursive=True):
            child.terminate()
        self.process.terminate()
        self.process.wait()
        self.__ports_in_use.remove(self.port)


class JenaQuery:

    def __init__(self):
        self.jena_folder = Path(os.environ.get(ENV_VAR_JENA_PATH)).resolve()
        if not self.jena_folder.exists():
            raise FileNotFoundError(f"Jena folder not found: {self.jena_folder}")

        if os.name == 'nt':
            self.jena_tdb_path = self.jena_folder.joinpath('bat', 'tdbquery.bat')
            self.shell = True
        elif os.name == 'posix':
            self.jena_tdb_path = self.jena_folder.joinpath('bin', 'tdbquery')
            self.shell = False
        else:
            raise OSError(f"Unsupported OS: {os.name}")

        full_script_path = self.jena_folder.joinpath(self.jena_tdb_path).resolve()
        if not full_script_path.exists():
            raise FileNotFoundError(f"Jena query script not found: {full_script_path}")

    def run_query(self, graph_path: Path, query: str) -> dict:
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph path not found: {graph_path}")

        fd, temp_query_path = tempfile.mkstemp(suffix='.sparql')
        try:
            # Write the query to the temporary file
            with os.fdopen(fd, 'w') as temp_query_file:
                temp_query_file.write(query)

            cmd_list = [
                str(self.jena_tdb_path),
                f'--mem={graph_path}',
                f'--query={temp_query_path}',
                f'--results=JSON',
            ]
            process = subprocess.run(
                cmd_list,
                cwd=self.jena_folder,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=self.shell,
                text=True,
            )
            out_str, out_err = process.stdout, process.stderr
            if out_err:
                raise RuntimeError(f"Error running query on shell: {out_err.strip()}")
            if not out_str:
                raise ValueError(f"Empty output from shell")
            result_json = json.loads(out_str)
        except Exception as e:
            raise e
        finally:
            os.unlink(temp_query_path)

        return result_json


if __name__ == '__main__':
    # rdf_file = Path('C:/Users/sandr/Desktop/Coding/KGRelated/LLMsforNL2SPARQL/datasets/spider4sparql/processed/graph/dev/car_1.rdf')
    # local_db = LocalFusekiServer(rdf_file, timeout=5)
    # local_db.stop()
    # query_engine = JenaQuery()
    # output = query_engine.run_query(rdf_file, 'select * where { ?s ?p ?o } limit 1')
    file = Path('/home/unica/Progetti/LLMs-for-SPARQL/datasets/spider4sparql/processed/graph/dev/battle_death.rdf')
    query = 'select * where {?s ?p ?o} limit 1'
    query_engine = JenaQuery()
    output = query_engine.run_query(file, query)
    pass
