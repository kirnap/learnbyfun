"""
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.
"""

# Definition for singly-linked list.


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

        
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        # create dummy head this is needed 'cuz don't have head/current in ListNode
        dh = ListNode(0)
        head = dh
        a1 = l1
        a2 = l2
        carry = 0
        while((a1 is not None) or (a2 is not None)):
            x = a1.val if a1 is not None else 0
            y = a2.val if a2 is not None else 0
            tot = carry + x + y
            head.next = ListNode(tot % 10)
            head = head.next
            carry = tot // 10
            if a1 is not None:
                a1 = a1.next
            if a2 is not None:
                a2 = a2.next
        if carry > 0:
            head.next = ListNode(carry)
        return dh.next
        

if __name__ == '__main__':
    l0 = ListNode(val=3, next=None)
    l1 = ListNode(val=4, next=l0)
    l6 = ListNode(val=6, next=None)
    l63 = ListNode(val=3, next=l6)
    l342 = ListNode(val=2, next=l1)
    l7 = ListNode(7)
    l87 = ListNode(0, next=l7)
    l5 = ListNode(5)
    l465 = ListNode(4, next=None)
    l65 = ListNode(6, next=l465)
    l5 = ListNode(5, next=l65) # 465
    t = Solution()
    sol = t.addTwoNumbers(l63,l342)  # 342 + 63 = 405
    assert sol.val == 5
    assert sol.next.val == 0
    assert sol.next.next.val == 4
    sol = t.addTwoNumbers(l5,l342)  # 342 + 465 = 807
    assert sol.val == 7
    assert sol.next.val == 0
    assert sol.next.next.val == 8

