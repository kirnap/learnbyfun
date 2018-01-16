;; customize interface

;;;;; linum related ;;;;;
;; prog-mode-hook contains all the programming modes for emacs global setting for all of them 
(add-hook 'prog-mode-hook 'linum-mode)
(global-linum-mode 1);; fix not hooked julia in prog-mode-hook 
(setq column-number-mode t)
(setq linum-format "%d ")

;; Julia mode related
;; TODO: color-theme not seeem so good fix it 
(add-to-list 'load-path "/KUFS/scratch/okirnap/.emacs.d/julia-emacs")
(require 'julia-mode)

;;;;;;;;;;;;;;;;;; Customization of keys ;;;;;;;;;;;;;;;;;;

;; Useful links:
;; https://stackoverflow.com/questions/9688748/emacs-comment-uncomment-current-line
;; http://ergoemacs.org/emacs/emacs_useful_user_keybinding.html

;; That binding defines a short-cut for a line beginning and line end 
(global-set-key 
 (kbd "C-x C-i")
 (lambda ()
   (interactive)
   (comment-or-uncomment-region (line-beginning-position) (line-end-position))))

;; That defines a short-cut for region beginning and region end
(global-set-key 
 (kbd "C-x C-k")
 (lambda ()
   (interactive)
   (comment-or-uncomment-region (region-beginning) (region-end))))
