;; You need to define custom function here and then you may include if from your init.el file

;;; Here is an example copy-line function originally taken from

;; https://www.emacswiki.org/emacs/CopyingWholeLines
;; After defining that function you need to include it by `load-file path/to/your-file`
(defun copy-line (arg)
  "Copy lines (as many as prefix argument) in the kill ring.
      Ease of use features:
      - Move to start of next line.
      - Appends the copy on sequential calls.
      - Use newline as last char even on the last line of the buffer.
      - If region is active, copy its lines."
  (interactive "p")
  (let ((beg (line-beginning-position))
	(end (line-end-position arg)))
    (when mark-active
      (if (> (point) (mark))
	  (setq beg (save-excursion (goto-char (mark)) (line-beginning-position)))
	(setq end (save-excursion (goto-char (mark)) (line-end-position)))))
    (if (eq last-command 'copy-line)
	(kill-append (buffer-substring beg end) (< end beg))
      (kill-ring-save beg end)))
  (kill-append "\n" nil)
  (beginning-of-line (or (and arg (1+ arg)) 2))
  (if (and arg (not (= 1 arg))) (message "%d lines copied" arg)))

;; in order to add keybinding for your specific function:
(global-set-key (kbd "C-c a") 'copy-line) ;; sets the short-cut
;; Check the following repo for more detailed function notes
;; https://github.com/hypernumbers/learn_elisp_the_hard_way/blob/master/contents/lesson-4-2-adding-custom-functions-to-emacs.rst
