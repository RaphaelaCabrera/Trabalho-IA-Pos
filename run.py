import argparse

from pipeline.pipeline import Pipeline


class Run:
    @staticmethod
    def parse_commands():
        parser = argparse.ArgumentParser(usage='%(prog)s task',
                                         description='Executar o pipeline desejado')

        parser.add_argument('task',
                            choices=['dataset1', 'dataset2', 'dataset3', 'dataset4', 'dataset5'],
                            help='Especifica qual dataset a ser utilizado e executa o seu respectivo pipeline')

        args = vars(parser.parse_args())
        return args['task']

    def execute(self):
        task = self.parse_commands()
        if task == 'dataset1':
            Pipeline().execute('dataset1')
        elif task == 'dataset2':
            Pipeline().execute('dataset2')
        elif task == 'dataset3':
            Pipeline().execute('dataset3')
        elif task == 'dataset4':
            Pipeline().execute('dataset4')
        elif task == 'dataset5':
            Pipeline().execute('dataset5')


if __name__ == '__main__':
    Run().execute()
