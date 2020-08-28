import random
import time
import numpy as  np
class Training:
    def __init__(self):
        P = 5
        R = 3
        end_time = time.time() + 5
        list = []
        slist = []
        while (time.time() < end_time):
            def calculateNeed(need, maxm, allot):
                # Calculating Need of each P
                for i in range(P):
                    for j in range(R):
                        need[i][j] = maxm[i][j] - allot[i][j]

            def isSafe(avail, maxm, allot):
                need = []
                for i in range(P):
                    l = []
                    for j in range(R):
                        l.append(0)
                    need.append(l)

                    # Function to calculate need matrix
                calculateNeed(need, maxm, allot)

                # Mark all processes as infinish
                finish = [0] * P

                # To store safe sequence
                safeSeq = [0] * P

                # Make a copy of available resources
                work = [0] * R
                for i in range(R):
                    work[i] = avail[i]

                    # While all processes are not finished
                # or system is not in safe state.
                count = 0
                while (count < P):

                    flag = False
                    for p in range(P):

                        # First check if a process is finished,
                        # if no, go for next condition
                        if (finish[p] == 0):

                            for j in range(R):
                                if (need[p][j] > work[j]):
                                    break

                            # If all needs of p were satisfied.
                            if (j == R - 1):

                                for k in range(R):
                                    work[k] += allot[p][k]

                                    # Add this process to safe sequence.
                                safeSeq[count] = p
                                count += 1

                                # Mark this p as finished
                                finish[p] = 1

                                flag = True

                    # If we could not find a next process
                    # in safe sequence.
                    if (flag == False):
                        print("System is not in safe state")
                        return 0

                print("System is in safe state.")
                return 1

            rlist = []
            for i in range(33):
                rlist.append(random.randint(1, 10))

                # Available instances of resources
                avail = [rlist[0], rlist[1], rlist[2]]

                # Maximum R that can be allocated to processes
                maxm = [[rlist[3], rlist[4], rlist[5]], [rlist[6], rlist[7], rlist[8]],
                        [rlist[9], rlist[10], rlist[11]], [rlist[12], rlist[13], rlist[14]],
                        [rlist[15], rlist[16], rlist[17]]]

                # Resources allocated to processes
                allot = [[rlist[18], rlist[19], rlist[20]], [rlist[21], rlist[22], rlist[23]],
                         [rlist[24], rlist[25], rlist[26]], [rlist[27], rlist[28], rlist[29]],
                         [rlist[30], rlist[31], rlist[32]]]

                # Check system is in safe state or not
                is_safe = isSafe(avail, maxm, allot)
                list.append(rlist)
                slist.append(is_safe)
        np.save("trained_data1", np.array(list))
        np.save("trained_data2", np.array(slist))
if __name__ == '__main__':
    Training()