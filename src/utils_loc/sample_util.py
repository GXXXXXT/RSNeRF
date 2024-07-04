import queue as Queue
import torch
import torch.nn as nn


class SortedDictQueue(nn.Module):
    def __init__(self):
        super().__init__()
        self.num = 0
        self.queue = Queue.PriorityQueue()

    def add_dict(self, dict_item):
        if dict_item['vertex'] is None:
            return
        key = dict_item['average']
        self.queue.put((-key, self.num, dict_item))
        self.num += 1

    def get_max_dict(self):
        if not self.queue.empty():
            return self.queue.get()[2]
        else:
            return None

    def get_queue_length(self):
        return self.queue.qsize()


def get_average_color(img):
    return ((img - img.mean()) ** 2).sum()


def get_center(matrix):
    return torch.tensor([matrix.shape[0] // 2, matrix.shape[1] // 2])


def get_neighbor_vertex(start, matrix, depth):
    depth = depth[start[0]:start[0] + matrix.shape[0], start[1]:start[1] + matrix.shape[1]]
    if torch.sum(depth > 0).item() == 0:
        return None
    index = get_center(matrix)
    if depth[index[0], index[1]] != 0:
        return index
    else:
        rows, cols = torch.where(depth > 0)
        min = torch.argmin(torch.sqrt((rows - index[0]) ** 2 + (cols - index[1]) ** 2))
        return torch.tensor([rows[min], cols[min]])


def quadtree(sample, depth):
    queue = SortedDictQueue()
    mid_x, mid_y = sample["matrix"].shape[0] // 2, sample["matrix"].shape[1] // 2
    if mid_y == 0 or mid_x == 0:
        sample["average"] = 0.0
        queue.add_dict(sample)
        return queue

    queue.add_dict({
        "start": sample["start"],
        "matrix": sample["matrix"][:mid_x, :mid_y],
        "average": get_average_color(sample["matrix"][:mid_x, :mid_y]),
        "vertex": get_neighbor_vertex(sample["start"], sample["matrix"][:mid_x, :mid_y], depth)
    })
    queue.add_dict({
        "start": sample["start"].add(torch.tensor([0, mid_y])),
        "matrix": sample["matrix"][:mid_x, mid_y:],
        "average": get_average_color(sample["matrix"][:mid_x, mid_y:]),
        "vertex": get_neighbor_vertex(sample["start"].add(torch.tensor([0, mid_y])), sample["matrix"][:mid_x, mid_y:], depth)
    })
    queue.add_dict({
        "start": sample["start"].add(torch.tensor([mid_x, 0])),
        "matrix": sample["matrix"][mid_x:, :mid_y],
        "average": get_average_color(sample["matrix"][mid_x:, :mid_y]),
        "vertex": get_neighbor_vertex(sample["start"].add(torch.tensor([mid_x, 0])), sample["matrix"][mid_x:, :mid_y], depth)
    })
    queue.add_dict({
        "start": sample["start"].add(torch.tensor([mid_x, mid_y])),
        "matrix": sample["matrix"][mid_x:, mid_y:],
        "average": get_average_color(sample["matrix"][mid_x:, mid_y:]),
        "vertex": get_neighbor_vertex(sample["start"].add(torch.tensor([mid_x, mid_y])), sample["matrix"][mid_x:, mid_y:], depth)
    })

    return queue


def get_sample_mask(img, depth, N_rays):
    H, W, B = img.shape
    mask, scope = torch.zeros((H, W)), torch.zeros((H, W))
    queue = SortedDictQueue()
    init = {
        "start": torch.tensor([0, 0]),
        "matrix": img,
        "average": get_average_color(img),
        "vertex": get_neighbor_vertex(torch.tensor([0, 0]), img, depth)
    }
    queue.add_dict(init)

    while queue.get_queue_length() < N_rays:
        ret = quadtree(queue.get_max_dict(), depth)
        while ret.get_queue_length() > 0 and queue.get_queue_length() < N_rays:
            queue.add_dict(ret.get_max_dict())

    pq = Queue.PriorityQueue()
    while queue.get_queue_length() > 0:
        dic = queue.get_max_dict()
        pq.put((dic["start"][0], dic["start"][1], dic))

    tmp = 0
    while pq.qsize() > 0:
        dic = pq.get()[2]
        scope[dic["start"][0]:dic["start"][0] + dic["matrix"].shape[0], dic["start"][1]:dic["start"][1] + dic["matrix"].shape[1]] = tmp
        tmp += 1
        index = dic["start"].add(dic["vertex"])
        # print(index[0], index[1], dic["average"])
        # img = cv2.circle(img, (int(index[1]), int(index[0])), 1, (0, 0, 255), -1)
        mask[index[0], index[1]] = 1
        # cv2.rectangle(img, (int(dic["start"][1]), int(dic["start"][0])), (int(dic["start"][1]) + dic["matrix"].shape[1], int(dic["start"][0]) + dic["matrix"].shape[0]), (1, 0, 0), 10)
        cv2.circle(img, (int(index[1]), int(index[0])), 1, (1, 0, 0), 40)
        # scope[dic["start"][0]:dic["start"][0] + dic["matrix"].shape[0], dic["start"][1]:dic["start"][1] + dic["matrix"].shape[1]] = torch.from_numpy(img[int(index[0]), int(index[1])]).unsqueeze(0).unsqueeze(0).expand(dic["matrix"].shape[0], dic["matrix"].shape[1], 3)
    mask = mask.type(torch.bool).cpu()
    scope = scope.cpu()
    return mask, scope


def sampling_without_replacement(logp, k):
    def gumbel_like(u):
        return -torch.log(-torch.log(torch.rand_like(u) + 1e-7) + 1e-7)

    scores = logp + gumbel_like(logp)
    return scores.topk(k, dim=-1)[1]


def sample_rays(mask, num_samples):
    B, H, W = mask.shape
    probs = mask / (mask.sum() + 1e-7)
    flatten_probs = probs.reshape(B, -1)
    sampled_index = sampling_without_replacement(
        torch.log(flatten_probs + 1e-7), num_samples)
    sampled_masks = (torch.zeros_like(
        flatten_probs).scatter_(-1, sampled_index, 1).reshape(B, H, W) > 0)
    return sampled_masks


if __name__ == '__main__':
    import cv2
    import torch

    img = cv2.imread("/Volumes/MacExtHD/Code/Python/Road-SLAM/data/apollo/Record001/img/170927_063817699_Camera_5.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) / 255.0
    # img = torch.from_numpy(img / 255.0).float()
    depth = cv2.imread("/Volumes/MacExtHD/Code/Python/Road-SLAM/data/apollo/Record001/depth/170927_063817699_Camera_5.png", cv2.IMREAD_ANYDEPTH)
    # depth = torch.from_numpy(depth).float()[:, :, 0]
    depth = torch.from_numpy(depth).float()
    mask, scope = get_sample_mask(img, depth, 128)
    # mask = sample_rays(torch.where(depth > 0,
    #                                torch.ones_like(depth)[None, ...],
    #                                torch.zeros_like(depth)[None, ...]),
    #                                256)[0, ...]
    # indices = torch.where(mask > 0)
    # for i, j in zip(indices[0].tolist(), indices[1].tolist()):
    #     cv2.circle(img, (j, i), 1, (1, 0, 0), 40)
    # for i in torch.linspace(0, img.shape[0], 16).tolist():
    #     for j in torch.linspace(0, img.shape[1], 16).tolist():
    #         cv2.circle(img, (int(j), int(i)), 1, (1, 0, 0), 40)
    cv2.imshow("img", cv2.cvtColor(cv2.convertScaleAbs(img * 255.0), cv2.COLOR_BGR2RGB))
    cv2.imwrite("/Users/lyjj/Downloads/128.png", cv2.cvtColor(cv2.convertScaleAbs(img * 255.0), cv2.COLOR_BGR2RGB))
    # cv2.imshow("scope", scope.numpy())
    cv2.waitKey(0)
