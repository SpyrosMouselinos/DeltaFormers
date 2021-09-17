import os

import neptune.new as neptune
from neptune.new.exceptions import NeptuneInvalidApiTokenException, FileUploadError

# Look for API TOKEN in the MACHINE
API_TOKEN = os.getenv('NEPTUNE_API_TOKEN')


class DummySession:
    def __init__(self):
        return

    def upload(self, *args, **kwargs):
        return


class Deltalogger:
    def __init__(self, project_name, run_tag=None, dummy=False):
        self.dummy = dummy
        self.run_tag = run_tag
        self.workspace = 'spyrosmouselinos'
        self.project_name = project_name
        if not self.dummy:
            try:
                self.session = neptune.init(project=f'{self.workspace}/{project_name}', api_token=API_TOKEN)
            except NeptuneInvalidApiTokenException:
                print(f"Connection to Neptune failed... check your API_TOKEN!\nAPI TOKEN loaded: {API_TOKEN}")

            if self.run_tag is not None:
                self.session["sys/tags"].add([str(f) for f in run_tag])
        else:
            self.session = {'rl_agent': DummySession()}
        return

    def store(self, args):
        if self.dummy:
            return
        if isinstance(args, dict):
            for key, value in args.items():
                self.session[key] = value
        elif isinstance(args, tuple):
            self.session[args[0]] = args[1]

    def log(self, args):
        if self.dummy:
            return
        if isinstance(args, dict):
            for key, value in args.items():
                self.session[key].log(value)
        elif isinstance(args, tuple):
            self.session[args[0]].log(args[1])

    def image_store(self, storage_name, file):
        if self.dummy:
            return
        file = neptune.types.File.as_image(file)
        try:
            self.session[storage_name].upload(file)
        except FileUploadError:
            print("Upload Unsuccessful :(\n")
        return

    def close(self):
        if self.dummy:
            return
        self.session.stop()

    def __repr__(self):
        if self.dummy:
            return
        print(f"Neptune AI Logger, currently operating on {self.project_name} experiment!\n")

    def __enter__(self):
        if self.dummy:
            return
        print("Note that you are using Deltalogger as context manager!\n This is not the intended use!")
        return self

    def __exit__(self, *args):
        if self.dummy:
            return
        self.session.stop()
        return
