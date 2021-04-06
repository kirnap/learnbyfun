# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from collections import deque
class Solution:
    def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
        ret = []
        node_queue = [root, None]
        level_items = deque([])
        toleft = True
        if root is None:
            return []

        while len(node_queue) > 0:
            node = node_queue.pop(0)
            if node:
                if toleft:
                    level_items.append(node.val)
                else:
                    level_items.appendleft(node.val)

                if node.left:
                    node_queue.append(node.left)
                if node.right:
                    node_queue.append(node.right)
            else: # level change
                ret.append(level_items)
                level_items = deque()
                toleft = not toleft
                if len(node_queue) > 0: # there are remaining items in the list
                    node_queue.append(None)
        return ret