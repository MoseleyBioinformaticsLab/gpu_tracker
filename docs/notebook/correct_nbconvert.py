with open('tutorial.rst', 'r') as file:
    contents: str = file.read()

contents: str = contents.replace('ipython3', 'python3')
contents: str = contents.replace('python3\n\n    !', 'none\n\n    !')
contents: str = contents.replace('parsed-literal::', 'code:: none')
contents: str = contents.replace('!', '$ ')
contents: str = contents.replace('os._exit(0)', '')
contents = '.. _tutorial-label:\n\n' + contents

with open('tutorial.rst', 'w') as file:
    file.write(contents)
