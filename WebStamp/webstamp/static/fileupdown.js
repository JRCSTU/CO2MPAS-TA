/*global saveAs, self*/

"use strict";
$(document).ready(function(event){
    var dice_report = $("#dice_report");
    var download_label = $("#download-label");
    var download_filename = $("#download-filename");
    var download_btn = $("#download-btn");

    if (!dice_report[0].readOnly) {
        download_label.addClass('hidden');
        download_filename.addClass('hidden');
        download_btn.addClass('hidden');
    } else {
        download_btn.click(function(event){
            saveAs(
                new self.Blob(
                    [dice_report.value],
                    {type: "text/plain;charset=" + document.characterSet}
                ),
                (download_filename.value || download_filename[0].placeholder || "stamp.txt")
            );
        });
    }
});
