import subprocess
import types

def run(files, args, fashion=False):
    for i, file in enumerate(files):
        subprocess.run('clear')
        print("[%d/%d] %s >> %s\n" % (i+1, len(files), file['name'], file['log_dir']))
        subprocess.run(['python', file['name'], ]
                     + ['--log_dir', file['log_dir']]
                     + ['--batch_size', str(args.batch_size)]
                     + ['--n_epochs', str(args.n_epochs)]
                     + ['--log_interval', str(args.log_interval)]
                     + (['--fashion'] if fashion else [])
        )

if __name__ == "__main__":

    args = types.SimpleNamespace()
    args.batch_size = 32
    args.n_epochs = 50
    args.log_interval = 5

    files1 = []
    files1.append(dict(name='tf_gan_mnist_mlp.py', log_dir='tf_gan_mnist_mlp'))
    files1.append(dict(name='tf_cgan_mnist_mlp.py', log_dir='tf_cgan_mnist_mlp'))
    files1.append(dict(name='tf_acgan_mnist_mlp.py', log_dir='tf_acgan_mnist_mlp'))

    files2 = []
    files2.append(dict(name='tf_gan_mnist_mlp.py', log_dir='tf_gan_fashion-mnist_mlp'))
    files2.append(dict(name='tf_cgan_mnist_mlp.py', log_dir='tf_cgan_fashion-mnist_mlp'))
    files2.append(dict(name='tf_acgan_mnist_mlp.py', log_dir='tf_acgan_fashion-mnist_mlp'))

    run(files1, args, fashion=False)
    run(files2, args, fashion=True)

