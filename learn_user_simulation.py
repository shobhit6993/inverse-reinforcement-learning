from imitation_learning.irl import IRL


def main():
    """Executes the IRL algorithm for building a user simulation."""
    irl = IRL()
    irl.run_irl()


if __name__ == '__main__':
    main()
