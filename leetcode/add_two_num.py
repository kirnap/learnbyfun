# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dh = ListNode(0) # create dummy head
        carry = 0
        p = l1
        q = l2
        curr = dh
        while((p != None) or (q != None)):
            x = p.val if p else 0
            y = q.val if q else 0
            tot = carry + x + y
            carry = int(tot / 10)
            curr.next = ListNode(tot % 10)
            curr = curr.next
            if p is not None:
                p = p.next
            if q is not None:
                q = q.next
        if (carry > 0):
            curr.next = ListNode(carry)
        return dh.next
        
    
