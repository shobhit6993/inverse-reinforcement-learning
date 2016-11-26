from imitation_learning.irl import IRL


def main():
    irl = IRL()
    expert_fe = irl.calc_feature_expectation(irl.real_user)
    sim_fe = irl.calc_feature_expectation(irl.sim_user)


if __name__ == '__main__':
    main()
