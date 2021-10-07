# solution of binary inorder traversal problem

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution1:  # recursive solution
    def __init__(self):
        pass

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        def helper(treenode, lst):
            if treenode is None:
                 return
            if treenode.left is not None:
                helper(treenode.left, lst)
            lst.append(treenode.val)
            if treenode.right is not None:
                helper(treenode.right, lst)
        ret = []
        helper(root, ret)
        return ret


class Solution2:
    # Set current to root of binary tree
    def __init__(self):
        pass

    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        current = root
        stack = []  # initialize stack
        done = 0
        ret = []
        while True:

            # Reach the left most Node of the current Node
            if current is not None:

                # Place pointer to a tree node on the stack
                # before traversing the node's left subtree
                stack.append(current)

                current = current.left


            # BackTrack from the empty subtree and visit the Node
            # at the top of the stack; however, if the stack is
            # empty you are done
            elif (stack):
                current = stack.pop()
                ret.append(current.val)
                # We have visited the node and its left
                # subtree. Now, it's right subtree's turn
                current = current.right

            else:
                return ret


