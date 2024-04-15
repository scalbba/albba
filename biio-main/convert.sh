
check(){
    file_path=$1
    mat_name=${file_path##*/}
    mat_name=${mat_name%.*}
    parent_name=${file_path%/*}
    parent_name=${parent_name##*/}
    #echo $parent_name
    #echo $mat_name
    if [ "$parent_name" = "$mat_name" ]; then
    return 0 # 0 eq true
    else
        return 1
    fi
}

convert_all(){
    for data in `find $1 -name "*.mtx"`
    do
        if check $data; then
            ./bin/convert $data
            rm -f $data
            echo -n "[SYST]  "
            echo -n $data
            echo " has been removed"
        fi
    done
}

data_path="../MM"

convert_all $data_path
