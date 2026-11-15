import time
import numpy as np
from scipy.spatial.transform import Rotation as R


class KinematicSolver:
    def __init__(self):
        # Initial rotation matrix - from H0 to H1
        rx = R.from_euler('x', 0, degrees=True)
        ry = R.from_euler('y', 0, degrees=True)
        rz = R.from_euler('z', -30, degrees=True)
        r = rz * ry * rx
        self.init_rotation_matrix = r.as_matrix()
        # self.init_translation_vector = np.array([0.0, 0.0, 234.756])
        self.init_translation_vector = np.array([0.0, 0.0, 256.792])    # With Ball joint
        # self.init_translation_vector = np.array([0.0, 0.0, 238.756])    # With new bearings
        self.init_T_1_0 = self.get_transform_matrix(self.init_rotation_matrix, self.init_translation_vector)
        # self.L = np.array([236.5, 236.5, 236.5, 236.5, 236.5, 236.5]).reshape(-1, 1) # was 236.5
        self.L = np.array([260.0, 260.0, 260.0, 260.0, 260.0, 260.0]).reshape(-1, 1) # was 236.5

        self.s1 = np.array([0, 0, 1]).reshape(-1, 1)
        self.s2 = np.array([0, 0, 1]).reshape(-1, 1)
        self.s3 = np.array([0, 0, 1]).reshape(-1, 1)
        self.s4 = np.array([0, 0, 1]).reshape(-1, 1)
        self.s5 = np.array([0, 0, 1]).reshape(-1, 1)
        self.s6 = np.array([0, 0, 1]).reshape(-1, 1)
        self.s_i = np.concatenate((self.s1, self.s2, self.s3, self.s4, self.s5, self.s6), axis=1)

        # self.S_0_1 = np.array([47.040, 104.524, 0.00]).reshape(-1, 1)
        # self.S_0_2 = np.array([114.040, -11.524, 0.00]).reshape(-1, 1)
        # self.S_0_3 = np.array([67.000, -93.000, 0.00]).reshape(-1, 1)
        # self.S_0_4 = np.array([-67.000, -93.000, 0.00]).reshape(-1, 1)
        # self.S_0_5 = np.array([-114.040, -11.524, 0.00]).reshape(-1, 1)
        # self.S_0_6 = np.array([-47.040, 104.524, 0.00]).reshape(-1, 1)
        # self.init_S_0 = np.concatenate((self.S_0_1, self.S_0_2, self.S_0_3, self.S_0_4, self.S_0_5, self.S_0_6), axis=1)
        # self.init_S_0 = self.init_S_0

        self.S_0_1 = np.array([57.606, 110.624, 0.00]).reshape(-1, 1)
        self.S_0_2 = np.array([124.606, -5.424, 0.00]).reshape(-1, 1)
        self.S_0_3 = np.array([67.000, -105.200, 0.00]).reshape(-1, 1)
        self.S_0_4 = np.array([-67.000, -105.200, 0.00]).reshape(-1, 1)
        self.S_0_5 = np.array([-124.606, -5.424, 0.00]).reshape(-1, 1)
        self.S_0_6 = np.array([-57.606, 110.624, 0.00]).reshape(-1, 1)
        self.init_S_0 = np.concatenate((self.S_0_1, self.S_0_2, self.S_0_3, self.S_0_4, self.S_0_5, self.S_0_6), axis=1)
        self.init_S_0 = self.init_S_0

        # self.P_1_1 = np.array([20.000, 95.000, 0.000]).reshape(-1, 1)
        # self.P_1_2 = np.array([92.272, -30.179, 0.000]).reshape(-1, 1)
        # self.P_1_3 = np.array([72.272, -64.821, 0.000]).reshape(-1, 1)
        # self.P_1_4 = np.array([-72.272, -64.821, 0.000]).reshape(-1, 1)
        # self.P_1_5 = np.array([-92.272, -30.179, 0.000]).reshape(-1, 1)
        # self.P_1_6 = np.array([-20.000, 95.000, 0.000]).reshape(-1, 1)
        # self.init_P_1 = np.concatenate((self.P_1_1, self.P_1_2, self.P_1_3, self.P_1_4, self.P_1_5, self.P_1_6), axis=1)
        # self.init_P_1 = self.init_P_1

        self.P_1_1 = np.array([-30.180, 92.272, 0.000]).reshape(-1, 1)
        self.P_1_2 = np.array([95.000, 20.000, 0.000]).reshape(-1, 1)
        self.P_1_3 = np.array([95.000, -20.000, 0.000]).reshape(-1, 1)
        self.P_1_4 = np.array([-30.180, -92.272, 0.000]).reshape(-1, 1)
        self.P_1_5 = np.array([-64.820, -72.272, 0.000]).reshape(-1, 1)
        self.P_1_6 = np.array([-64.820, 72.272, 0.000]).reshape(-1, 1)
        self.init_P_1 = np.concatenate((self.P_1_1, self.P_1_2, self.P_1_3, self.P_1_4, self.P_1_5, self.P_1_6), axis=1)
        self.init_P_1 = self.init_P_1

        self.P_0_1 = self.transform_1_0(self.init_T_1_0, self.P_1_1)
        self.P_0_2 = self.transform_1_0(self.init_T_1_0, self.P_1_2)
        self.P_0_3 = self.transform_1_0(self.init_T_1_0, self.P_1_3)
        self.P_0_4 = self.transform_1_0(self.init_T_1_0, self.P_1_4)
        self.P_0_5 = self.transform_1_0(self.init_T_1_0, self.P_1_5)
        self.P_0_6 = self.transform_1_0(self.init_T_1_0, self.P_1_6)
        self.init_P_0 = np.concatenate((self.P_0_1, self.P_0_2, self.P_0_3, self.P_0_4, self.P_0_5, self.P_0_6), axis=1)
        self.init_P_0 = np.resize(self.init_P_0, (3, self.init_P_0.shape[1]))

        self.IK_Solver_Duration = None

    @staticmethod
    def get_transform_matrix(rotation_matrix, translation_vector):
        translation_vector = np.reshape(translation_vector, (3, 1))
        transform = np.concatenate((rotation_matrix, translation_vector), axis=1)
        transform = np.concatenate((transform, np.array([[0, 0, 0, 1]])))
        return transform

    @staticmethod
    def transform_1_0(T_1_0, T):
        if T.shape[0] < 4:
            T = np.concatenate((T, (np.array([1]).reshape(-1, 1))))
        transform = np.matmul(T_1_0, T)
        return transform

    def calc_S(self, q):
        S = np.zeros(self.init_S_0.shape)
        S = self.init_S_0 + q.T * self.s_i
        # for i in range(q.shape[0]):
        #     S[:, i] = self.init_S_0[:, i] + q[i] * self.s_i[:, i]
        return S

    def calc_q(self, pose, rot, P):
        start = time.time()
        pose = pose.reshape(-1, 1)
        a = pose + np.matmul(rot, P) - self.init_S_0
        q1 = []
        q2 = []
        for i in range(6):
            q1.append(np.matmul(self.s_i[:, i].T, a[:, i]) +
                      np.sqrt(np.power(np.matmul(self.s_i[:, i].T, a[:, i]), 2) -
                              np.matmul(a[:, i].T, a[:, i]) +
                              np.power(self.L[i], 2)))
            q2.append(np.matmul(self.s_i[:, i].T, a[:, i]) -
                      np.sqrt(np.power(np.matmul(self.s_i[:, i].T, a[:, i]), 2) -
                              np.matmul(a[:, i].T, a[:, i]) +
                              np.power(self.L[i], 2)))
        q1 = np.array(q1, dtype=np.double)
        q2 = np.array(q2, dtype=np.double)
        self.IK_Solver_Duration = time.time() - start
        return q1, q2

    def discretize(self, delta_x, T, step_size=1):
        x = []
        q1 = []
        q2 = []
        delta_x = np.array(delta_x)
        for i, t in enumerate(range(0, T, step_size)):
            z = float(t / T)
            # x.append(((-20.0*z**7)+(70.0*z**6)-(84.0*z**5)+(35.0*z**4))*delta_x)
            x.append((((3 * z ** 2) - (2 * z ** 3)) * delta_x))  # Cubic Polynomial
            pose_1 = np.array([x[i][0], x[i][1], x[i][2]]).reshape(-1, 1)
            rx = R.from_euler('x', x[i][3], degrees=True)
            ry = R.from_euler('y', x[i][4], degrees=True)
            rz = R.from_euler('z', x[i][5], degrees=True)
            r = rz * ry * rx
            rot = r.as_matrix()
            transform = self.get_transform_matrix(rot, pose_1)
            T_1_0 = self.transform_1_0(self.init_T_1_0, transform)
            pose_0 = T_1_0[:, -1][:3]
            q1_temp, q2_temp = self.calc_q(pose_0, rot, self.init_P_1)
            q1.append(q1_temp)
            q2.append(q2_temp)
        x = np.array(x)
        q1 = np.array(q1)
        q2 = np.array(q2)
        return x, q1, q2

    def calc_X(self, X_hat, Q_given, delta_X_mm=2, delta_X_degrees=5, epsilon=0.01, n_iter=30):
        X_hat = np.array(X_hat) # First estimate of the FK solution (TCP coordinates)
        for i in range(n_iter):
            print('--- Iteration', str(i), '---')
            # Calculate all Q in the direct proximity of current estimated solution X
            X_grid = {}
            for x in list([X_hat[0]-delta_X_mm, X_hat[0], X_hat[0]+delta_X_mm]):
                for y in list([X_hat[1]-delta_X_mm, X_hat[1], X_hat[1]+delta_X_mm]):
                    for z in list([X_hat[2]-delta_X_mm, X_hat[2], X_hat[2]+delta_X_mm]):
                        for t_x in list([X_hat[3]-delta_X_degrees, X_hat[3], X_hat[3]+delta_X_degrees]):
                            for t_y in list([X_hat[4]-delta_X_degrees, X_hat[4], X_hat[4]+delta_X_degrees]):
                                for t_z in list([X_hat[5]-delta_X_degrees, X_hat[5], X_hat[5]+delta_X_degrees]):
                                    X_test_point = (x,y,z,t_x,t_y,t_z)
                                    pose_test = X_test_point[:3]
                                    rx = R.from_euler('x', X_test_point[3], degrees=True)
                                    ry = R.from_euler('y', X_test_point[4], degrees=True)
                                    rz = R.from_euler('z', X_test_point[5], degrees=True)
                                    r = rz * ry * rx
                                    rot_test = r.as_matrix()
                                    transform = self.get_transform_matrix(rot_test, pose_test)
                                    T_1_0_f = self.transform_1_0(self.init_T_1_0, transform)
                                    pose_0_test = T_1_0_f[:, -1][:3]
                                    _, Q_test = self.calc_q(pose_0_test, rot_test, self.init_P_1)
                                    Q_test = np.array(Q_test)
                                    X_grid.update({X_test_point : Q_test})
            # Find X in the proximity with Q(X) closest to the Q_given
            for key, value in X_grid.items():
                # Rewrite Q on the grid with Euclidean distances of Q from Q_given
                X_grid[key] = np.linalg.norm(value-Q_given)
            X_hat = min(X_grid, key=X_grid.get)
            # Console output
            print('Current best solution X =', X_hat)
            print('Current best solution has epsilon =', X_grid[X_hat])
            # Divide delta_X grid size by 2
            delta_X_mm = delta_X_mm / 2
            delta_X_degrees = delta_X_degrees / 2
            if X_grid[X_hat] < epsilon:
                return X_hat
            else:
                continue


if __name__ == "__main__":
    solver = KinematicSolver()
    X_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    X_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])  # millimeters and rads
    pose_1_curr = X_1[:3]
    rx = R.from_euler('x', X_1[3], degrees=True)
    ry = R.from_euler('y', X_1[4], degrees=True)
    rz = R.from_euler('z', X_1[5], degrees=True)
    r = rz * ry * rx
    rot_curr = r.as_matrix()
    transform = solver.get_transform_matrix(rot_curr, pose_1_curr)
    T_1_0_f = solver.transform_1_0(solver.init_T_1_0, transform)
    pose_0_curr = T_1_0_f[:, -1][:3]
    _, q_curr = solver.calc_q(pose_0_curr, rot_curr, solver.init_P_1)
    q_curr = np.array(q_curr)
    print('Q_given:', q_curr)
    X_f = solver.calc_X(X_0, q_curr)
    print('Init X_i:', X_0)
    print('FK X_f:', X_f)
    print('Actual X_f:', X_1)