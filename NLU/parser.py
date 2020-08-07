
import os
import json
import datetime
import numpy as np
import pandas as pd


def get_dir():
    directory = r'C:\Users\peter ye\Desktop\School\Seniro Spring\NLP\Project\data\Transcripts_1'
    directory2 = r'C:\Users\peter ye\Desktop\School\Seniro Spring\NLP\Project\data\cleaned_transcript'
    raw_path = []
    cleaned_path = []
    company = []
    dates_obj = []
    dates_str = []
    for subdir, dirs, files in os.walk(directory):
        # get company name
        company_name = subdir.split("\\")[-1]
        

        for file in files:
            if file != 'desktop.ini':
    
                #get a list of original transcript path
                filepath = subdir + os.sep + file
                raw_path.append(filepath)
                company.append(company_name)
                
                #get the new path for parsed transcript
                newpath = directory2 + os.sep + company_name + os.sep + file
                cleaned_path.append(newpath)
                
                # create the path directory
                newdir = directory2 + os.sep + company_name
                try:
                    os.stat(newdir)
                except:
                    os.mkdir(newdir)
    
                #get date

                date = str(file).split('-')[2:]
                date[-1] = date[-1].split('T')[0]
                date_obj = datetime.date(int(date[0]), int(date[1]), int(date[2]))
                date_str = str(date[1].lstrip('0')) +'/' + str(date[2].lstrip('0')) +'/' + str(date[0])
    
                if len(date) != 3:
                    raise ValueError('this date is not parsed correctly: check on day number of files')
                
                dates_obj.append(date_obj)
                dates_str.append(date_str)
            
            
    if len(raw_path) != len(cleaned_path):
        raise ValueError('the two paths are not of equal lenth, check again.')
    
    return raw_path, cleaned_path, company, dates_str

def parse(lines):
    presentation = 0
    qa = 0

    presentation_content_lst = []
    qa_content_lst = []


    initials = ['Presentation', 'Transcript']
    for line in lines:
        if presentation == 1:
            try:
                line = line.split('-').remove('')
            except:
                pass
            if line is not None:
                presentation_content_lst.append(line)


        if line in initials:
            presentation = 1
        elif line == 'Questions and Answers':
            presentation = 0
            qa = 1
        elif line == 'ready for questions':
            presentation = 0
            qa = 1
        elif line == 'Definitions':
            qa = 0
            presentation = 0

        if qa == 1:
            try:
                line = line.split('-').remove('')
            except:
                pass
            if line is not None:
                qa_content_lst.append(line)
        

    whole_content_lst = presentation_content_lst + qa_content_lst 
    return qa_content_lst, presentation_content_lst, whole_content_lst


def divider(data, on, max_len):
    new_data = []
    for dic in data:

        try:

            text = dic['whole'].split(' ')
        
        except:
            raise ValueError('please select parameter on in [whole, presentation, qa]')
        corpus_index = 0
        seperator = ' '
        while len(text) >= max_len:

            sub_dic = {}
            sub_dic['ticker'] = dic['ticker']
            sub_dic['date'] = dic['date']
            sub_dic['ex_return'] = dic['ex_return']
            sub_dic['direction'] = dic['direction'] 
            sub_dic['corpus_index'] = corpus_index
            sub_dic[on] = seperator.join(text[:max_len])

            new_data.append(sub_dic)
            
            
            corpus_index += 1
            text = text[max_len:]


    return new_data
            

def main():
    raw_path, cleaned_path, company, dates = get_dir()


    labels = pd.read_csv(r'C:\Users\peter ye\Desktop\School\Seniro Spring\NLP\Project\data\labels.csv')

    data = []

    k= 0
    for i in range(len(raw_path)):

        label = labels[(labels['Ticker'] == company[i]) & (labels['Date'] == dates[i])]

        ex_return = label['Excess Return'].values
        direction = label['Direction'].values



        if len(ex_return) != 0:
            
        
            dic = {}
            

            f = open(raw_path[i], 'r', encoding ='latin')
            if f.mode == 'r':
                content = f.read()
                lines = content.split('\n')

        
            qa, present, whole = parse(lines)
            seperator = ' '
            dic['ticker'] = company[i]
            dic['date'] = dates[i]

            dic['qa'] = seperator.join(qa)
            dic['whole'] = seperator.join(whole)
            dic['presentation'] = seperator.join(present)
            dic['ex_return'] = ex_return[0]
            dic['direction'] = direction[0]

            f.close()
            k += 1
            
            print('progress: ', i, '/', len(raw_path)-1, '\n' )
            
            data.append(dic)
        
        '''
        if len(data) == 2:
            break
        '''
        
        
        new_data = divider(data, 'whole', 460)
        

    target = r'C:\Users\peter ye\Desktop\School\Seniro Spring\NLP\Project\data\single_file'
    try:
        os.remove(target)
    except:
        pass
    
    with open(target, 'w+') as outfile:
        json.dump(new_data, outfile)


main()



    
