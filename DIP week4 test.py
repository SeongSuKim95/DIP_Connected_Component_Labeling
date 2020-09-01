for i in range(Labeled_img.shape[0]):
    if img[i] != 0:
        if Labeled_img[i - x] != 0:
            Labeled_img[i] = Labeled_img[i - x]
        elif Labeled_img[i - 1] != 0:
            Labeled_img[i] = Labeled_img[i - 1]
            if Labeled_img[i - x + 1] != 0 and Labeled_img[i - x + 1] != Labeled_img[i - 1]:
                pair_list = np.append(pair_list, [Labeled_img[i - x + 1], Labeled_img[i - 1]])
        elif Labeled_img[i - x - 1] != 0:
            Labeled_img[i] = Labeled_img[i - x - 1]
            if Labeled_img[i - x + 1] != 0 and Labeled_img[i - x + 1] != Labeled_img[i - x - 1]:
                pair_list = np.append(pair_list, [Labeled_img[i - x + 1], Labeled_img[i - x - 1]])
        # elif Labeled_img[i-1][j+1] !=0 and (Labeled_img[i-1][j-1], Labeled_img[i-1][j], Labeled_img[i][j-1])==(0,0,0) :
        elif Labeled_img[i - x + 1] != 0 and Labeled_img[i - x - 1] == 0 and Labeled_img[i - x] == 0 and Labeled_img[
            i - 1] == 0:
            Labeled_img[i] = Labeled_img[i - x + 1]
        else:
            Labeled_img[i] = cnt
            cnt += 1
