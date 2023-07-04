Dear Technical Editor,

Thank you for handling the review process.

The code is under the follow folder in the virtual machine (VM),
    ~/Desktop/PWAS/
which is a git clone from the public repository of the package:
    https://github.com/mjzhu-p/PWAS

The package has been uploaded to PyPi and can be installed via
    pip install pwasopt

A detailed README.md file is available on the GitHub page.

The main files of the solver is under the folder `src`:
    ~/Desktop/PWAS/src/pwasopt


The examples for the package is available under the folder `examples`:
    ~/Desktop/PWAS/examples/

where, we also include two separate files to ease the benchmark testing process on the server:
    mixed_variable_benchmarks_all.py: which includes all the benchmarks reported in the paper (see Table 2-8)
        - benchmark: Func-2C, Func-3C, Ackley-cC, XG-MNIST, NAS-CIFAR10, Horst6_hs044_modified, roscam_modified
            - Note, for comparison with other solvers, the optimization result is reported as -fopt in Table 3-8 in the paper since MAXIMIZATION is used; while in PWAS/PWASp, we use MINIMIZATION.
        - Notes for benchmark 'NAS-CIFAR10': since the dataset is compiled with TensorFlow version 1.x, python version < 3.8 is required (with TensorFlow < 2.x)
            - Therefore, to avoid switching back and force between different versions of python, we created a virtual environment with Python 3.7 and included all the dependencies within this virtual environment

    other_benchmarks_all.py: include various NLP, MIP, INLP, MIP benchmarks tested with PWAS/PWASp


Following, we note the steps to test these benchmarks:
    1. Enter the virtual environment using the following code:
        source ~/virtual_env/venv_with_python3.7/bin/activate

    2. Go to the example folder within the PWAS pacakage:
        cd Desktop/PWAS/examples/

    3. Test benchmark using the following code:
        python3 [benchmark_file].py [benchmark_name] [solver] [savefig]

        here
            [benchmark_file] can be either mixed_variable_benchmarks_all or other_benchmarks_all
            for [benchmark_name],
                if [benchmark_file] == mixed_variable_benchmarks_all,
                    [benchmark_name] can be Func-2C, Func-3C, Ackley-cC, XG-MNIST, NAS-CIFAR10, Horst6_hs044_modified, roscam_modified

                elif [benchmark_file] == other_benchmarks_all,
                    [benchmark_name] can be Bunnag6, PWA_example, camelsixhumps, ackley, camelsixhumps-linearconstr_2, .... (see a complete list within the 'other_benchmarks_all.py' file

            set [solver] == 1 if run PWAS, [solver] == 0 if run PWASp

            set [savefig] == 1 if want to save the figure showing 'Best value of latent function' vs. 'number of fun. eval' (if run PWAS) or 'preference queries' (if run PWASp)


        Examples:
            test benchmark XG-MNIST with PWAS and save the figure:
                python3 mixed_variable_benchmarks_all.py XG-MNIST 1 1

            test benchmark NAS-CIFAR10 with PWASp and do not save the figure:
                python3 mixed_variable_benchmarks_all.py NAS-CIFAR10 0 0

            test benchmark Bunag6 with PWASp and save the figure:
                python3 other_benchmarks_all.py Bunnag6 0 1

    4. To test other benchmarks, one can follow the notes on the README.md on the GitHub page, or use the same template as in mixed_variable_benchmarks_all.py and other_benchmarks_all.py


Debugging tips:
    When test the benchmarks, if there is an error, double check if the [benchmark] is within the [benchmark_file]
        double check if the spelling of the [benchmark] is consistent with the one within the [benchmark_file]

    When test benchmark NAS-CIFAR10, be sure to be within the python 3.7 virtual environment



Best regards,
Mengjia Zhu

