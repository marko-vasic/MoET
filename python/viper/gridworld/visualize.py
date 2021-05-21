from viper.gridworld.environment import SIZE
import pickle as pk
import numpy as np
import sympy


def visualize_gate_by_sampling(moe):
    print("Gating (expert ids)")
    for y in range(SIZE - 1, -1, -1):
        print("y={}    ".format(y)),
        for x in range(SIZE):
            expert_id = moe.predict_expert([[x, y]])[0]
            print("{}    ".format(expert_id)),
        print("")
    print("")
    print("      "),
    for x in range(SIZE):
        print("x={}  ".format(x)),
    print("")


def show_gate_equations(moe):
    print("GATING EQUATIONS")

    experts = moe.dtc_list
    E = len(experts)
    F = moe.tetag.shape[0] / E
    gating = moe.tetag.reshape(E, F).T

    xP, yP = sympy.symbols("x' y'")
    featureM = sympy.Matrix([xP, yP, 1])
    gatingM = sympy.Matrix(gating)
    equations = gatingM.T.dot(featureM)
    print("Equations in normalized space.")
    print(equations)

    x, y = sympy.symbols("x y")
    xP = (x - moe.scaler.mean_[0]) / moe.scaler.scale_[0]
    yP = (y - moe.scaler.mean_[1]) / moe.scaler.scale_[1]
    featureM = sympy.Matrix([xP, yP, 1])
    equations = gatingM.T.dot(featureM)
    print("Equations in original space.")
    print(equations)

    print("Expert 0 is used when following equation is sattisfied")
    print(equations[0] - equations[1] > 0)

    # Calculation example
    z = np.array([0., 0.])
    z -= moe.scaler.mean_
    z /= moe.scaler.scale_
    # Append bias term.
    z = np.append(z, [1])

    experts_num = len(experts)
    features_num = 3
    option_values = [0., 0.]
    for i in range(experts_num):
        for j in range(features_num):
            option_values[i] += z[j] * gating[j][i]


def main():
    policy_path = '/home/UNK/projects/explainableRL/viper/data/experiments/gridworld/size_5/MOE/moe_policy_e2_d1.pk'
    with open(policy_path, "rb") as f:
        moe_policy = pk.load(f)

    moe = moe_policy.moe

    visualize_gate_by_sampling(moe)
    show_gate_equations(moe)

if __name__ == '__main__':
    main()
